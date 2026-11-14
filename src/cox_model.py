"""Two cause-specific Cox models and probability-conserving CIF prediction."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtr, logsumexp
from statsmodels.duration.hazard_regression import PHReg


@dataclass
class CoxFit:
    cause: int
    columns: list
    beta: np.ndarray
    covariance: np.ndarray
    baseline: pd.DataFrame
    log_likelihood: float
    max_abs_score: float
    warnings: list
    time_effects: tuple = ()

    @property
    def static_columns(self):
        terms = getattr(self,'time_effects',())
        return self.columns[:-len(terms)] if terms else self.columns

    def coefficient_table(self):
        se = np.sqrt(np.diag(self.covariance))
        result = pd.DataFrame({'cause':self.cause, 'variable':self.columns,
            'beta':self.beta, 'se':se, 'HR':np.exp(self.beta),
            'HR_low':np.exp(self.beta-1.96*se), 'HR_high':np.exp(self.beta+1.96*se),
            'p':2*ndtr(-np.abs(self.beta/se))})
        result['interpretation'] = ['time interaction per log(t/36)' if c.endswith('__log_time')
            else 'spline basis coefficient; use joint contrasts' if c.endswith('__rcs')
            else 'linear contrast at month 36' if c in getattr(self,'time_effects',())
            else 'linear coefficient conditional on other design terms' for c in self.columns]
        return result


def model_design(model, x, time):
    """Actual covariates at event/prediction time; never use loan exit as time."""
    static = x[model.static_columns] if isinstance(x,pd.DataFrame) else x
    a = np.asarray(static,float)
    effects = getattr(model,'time_effects',())
    if not effects:
        return a
    if time <= 0:
        raise ValueError('Time-dependent design requires t>0')
    indices = [model.static_columns.index(c) for c in effects]
    return np.column_stack([a,a[:,indices]*np.log(float(time)/36.)])


def reference_rows(encoder, n=1):
    """Raw median/reference profiles; missing indicators set to observed."""
    values = dict(encoder.medians)
    values.update({c:(float(r) if c in ['number_of_units','number_of_borrowers'] else r)
                   for c,r in encoder.references.items()})
    return pd.DataFrame([values]*n)


def hazard_contrasts(model, encoder, variable, values, time=36):
    """Joint log-hazard contrast versus training median, including shape/TV CI."""
    profiles = reference_rows(encoder,len(values))
    profiles[variable] = np.asarray(values,float)
    reference = reference_rows(encoder,len(values))
    delta = model_design(model,encoder.transform(profiles),time)-model_design(model,encoder.transform(reference),time)
    eta = delta@model.beta
    se = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',delta,model.covariance,delta),0))
    return pd.DataFrame({'value':values,'time':time,'HR':np.exp(eta),
                         'low':np.exp(eta-1.96*se),'high':np.exp(eta+1.96*se)})


def risk_moments(data, x, beta, cause, second=False):
    """Monthly sufficient statistics with exactly L<t<=T (Breslow ties)."""
    a = np.asarray(x, dtype=float)
    L, T, J = [data[c].to_numpy() for c in ['L','T','J']]
    weights = np.exp(a @ beta)
    assert np.isfinite(weights).all()
    for t in np.unique(T[J==cause]):
        risk = (L<t) & (T>=t)
        event = (T==t) & (J==cause)
        w, ar = weights[risk], a[risk]
        s0 = w.sum()
        mean = w @ ar/s0
        info = None
        if second:
            info = (ar.T*w)@ar/s0-np.outer(mean,mean)
        yield float(t), event, float(event.sum()), s0, mean, info


def fit_cause(data, x, cause):
    model = PHReg(data['T'].to_numpy(float), x.to_numpy(float),
                  status=data.J.eq(cause).to_numpy(int),
                  entry=data.L.to_numpy(float)+1e-7, ties='breslow', missing='raise')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = model.fit(disp=0, maxiter=150, tol=1e-9)
    beta, covariance = np.asarray(result.params), np.asarray(result.cov_params())
    assert np.isfinite(beta).all() and np.isfinite(covariance).all()
    score = model.score(beta)
    if np.max(np.abs(score)) > 1e-3:
        raise RuntimeError(f'Cause {cause} did not converge: score={score}')
    records = []
    manual_score = np.zeros(len(beta))
    for t, event, d, s0, mean, _ in risk_moments(data, x, beta, cause):
        records.append({'month':t,'dH0':d/s0,'events':d,'risk_weight':s0})
        manual_score += x.to_numpy()[event].sum(axis=0)-d*mean
    np.testing.assert_allclose(manual_score, score, atol=1e-6)
    baseline = pd.DataFrame(records)
    baseline['H0'] = baseline.dH0.cumsum()
    return CoxFit(cause, x.columns.tolist(), beta, covariance, baseline,
                  float(model.loglike(beta)), float(np.max(np.abs(score))),
                  [f'{type(w.message).__name__}: {w.message}' for w in caught])


def fit_models(train, x, columns_by_cause=None):
    return {k: fit_cause(train, x if columns_by_cause is None else x[columns_by_cause[k]], k)
            for k in [1,2]}


def time_risk_moments(data, x, beta, cause, effects, second=True):
    """Breslow moments for X beta + sum X_j gamma_j log(t/36), strict entry."""
    a = np.asarray(x,float)
    indices = [x.columns.get_loc(c) for c in effects]
    L,T,J = [data[c].to_numpy() for c in ['L','T','J']]
    for t in np.unique(T[J==cause]):
        risk,event = (L<t)&(t<=T),(T==t)&(J==cause)
        g = np.log(float(t)/36.)
        ar,ae = a[risk],a[event]
        wr = np.column_stack([ar,ar[:,indices]*g])
        we = np.column_stack([ae,ae[:,indices]*g])
        eta = wr@beta
        logden = logsumexp(eta)
        weights = np.exp(eta-logden)
        mean = weights@wr
        variance = (wr.T*weights)@wr-np.outer(mean,mean) if second else None
        yield float(t),event,risk,wr,we,weights,float(logden),mean,variance


def time_likelihood(data, x, beta, cause, effects, information=True):
    """Analytic log partial likelihood, score and observed information."""
    score=np.zeros(len(beta)); info=np.zeros((len(beta),len(beta))); ll=0.
    for _,event,_,_,we,_,logden,mean,variance in time_risk_moments(data,x,beta,cause,effects,information):
        deaths=int(event.sum())
        ll+=float((we@beta).sum()-deaths*logden)
        score+=we.sum(axis=0)-deaths*mean
        if information:info+=deaths*variance
    return ll,score,info


def fit_time_varying(data, x, cause, effects, initial=None):
    """Small analytic Newton fit, line search; no expansion into future-exit X."""
    effects=tuple(effects)
    if not effects or any(c not in x for c in effects):
        raise ValueError('Time effects must name existing static design columns')
    beta=np.r_[initial.beta if initial is not None else np.zeros(x.shape[1]),np.zeros(len(effects))]
    for iteration in range(80):
        ll,score,info=time_likelihood(data,x,beta,cause,effects)
        if np.max(np.abs(score))<1e-6:break
        step=np.linalg.solve(info,score)
        fraction=1.
        while fraction>1e-8:
            candidate=beta+fraction*step
            new_ll,_,_=time_likelihood(data,x,candidate,cause,effects,information=False)
            if new_ll>=ll-1e-9:break
            fraction*=.5
        if fraction<=1e-8:raise RuntimeError('Time-dependent Cox line search failed')
        beta=candidate
    ll,score,info=time_likelihood(data,x,beta,cause,effects)
    if np.max(np.abs(score))>1e-4 or np.linalg.eigvalsh(info).min()<=0:
        raise RuntimeError('Time-dependent Cox convergence/information failure')
    covariance=np.linalg.inv(info)
    rows=[]
    for t,event,_,_,_,_,logden,_,_ in time_risk_moments(data,x,beta,cause,effects,False):
        deaths=int(event.sum())
        rows.append({'month':t,'dH0':float(deaths*np.exp(-logden)),
                     'events':deaths,'risk_weight':float(np.exp(logden))})
    baseline=pd.DataFrame(rows);baseline['H0']=baseline.dH0.cumsum()
    return CoxFit(cause,[*x.columns,*[c+'__log_time' for c in effects]],beta,covariance,
                  baseline,ll,float(np.max(np.abs(score))),[],effects)


def predict_cif(models, x, horizons, start=0, method='exponential'):
    """Conditional F_k(start,t] using exponential allocation of joint hazard.

    start may be a scalar landmark or a per-loan entry time. Each baseline jump
    is viewed as a piecewise constant competing-hazard step. This gives finite,
    monotone probabilities without clipping the sum of two independent risks.
    """
    h = np.asarray(horizons, float)
    assert np.all(np.diff(h)>0)
    starts = np.broadcast_to(np.asarray(start, float), len(x))
    if np.any(h[None,:] < starts[:,None]):
        raise ValueError('Prediction horizon cannot precede conditioning time')
    if method not in ('exponential','direct'):
        raise ValueError('Unknown competing-hazard integration method')
    static_rate = {k:np.exp(model_design(models[k],x,36)@models[k].beta)
                   for k in [1,2] if not getattr(models[k],'time_effects',())}
    out = np.zeros((len(x), len(h), 3))
    F = np.zeros((len(x),2)); S = np.ones(len(x))
    baselines = {k:models[k].baseline.set_index('month').dH0 for k in [1,2]}
    times = np.union1d(baselines[1].index, baselines[2].index)
    pointer = 0
    for j, horizon in enumerate(h):
        while pointer < len(times) and times[pointer] <= horizon:
            t = times[pointer]
            dh = np.array([baselines[k].get(t,0.) for k in [1,2]])
            rate = np.column_stack([static_rate[k] if k in static_rate else np.exp(model_design(models[k],x,t)@models[k].beta)
                                    for k in [1,2]])
            assert np.isfinite(rate).all()
            individual = rate*dh*(starts<t)[:,None]
            total = individual.sum(axis=1)
            share = np.divide(individual, total[:,None], out=np.zeros_like(individual), where=total[:,None]>0)
            if method=='direct' and total.max()>1+1e-12:
                raise ValueError('Direct product-integral has a combined jump >1')
            q = -np.expm1(-total) if method=='exponential' else total
            F += (S*q)[:,None]*share
            S *= np.exp(-total) if method=='exponential' else 1-total
            pointer += 1
        out[:,j,:2], out[:,j,2] = F, S
    assert np.isfinite(out).all() and out.min() >= -1e-12 and out.max() <= 1+1e-12
    assert (np.diff(out[:,:,:2],axis=1) >= -1e-12).all()
    np.testing.assert_allclose(out.sum(axis=2),1.,atol=1e-10)
    return out
