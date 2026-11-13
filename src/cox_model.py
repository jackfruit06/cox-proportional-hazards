"""Two cause-specific Cox models and probability-conserving CIF prediction."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtr
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

    def coefficient_table(self):
        se = np.sqrt(np.diag(self.covariance))
        return pd.DataFrame({'cause':self.cause, 'variable':self.columns,
            'beta':self.beta, 'se':se, 'HR':np.exp(self.beta),
            'HR_low':np.exp(self.beta-1.96*se), 'HR_high':np.exp(self.beta+1.96*se),
            'p':2*ndtr(-np.abs(self.beta/se))})


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


def fit_models(train, x):
    return {k: fit_cause(train, x, k) for k in [1,2]}


def predict_cif(models, x, horizons, start=0):
    """Conditional F_k(start,t] using exponential allocation of joint hazard.

    start may be a scalar landmark or a per-loan entry time. Each baseline jump
    is viewed as a piecewise constant competing-hazard step. This gives finite,
    monotone probabilities without clipping the sum of two independent risks.
    """
    h = np.asarray(horizons, float)
    assert np.all(np.diff(h)>0)
    a = np.asarray(x, float)
    rate = np.column_stack([np.exp(a@models[k].beta) for k in [1,2]])
    assert np.isfinite(rate).all()
    starts = np.broadcast_to(np.asarray(start, float), len(x))
    out = np.zeros((len(x), len(h), 3))
    F = np.zeros((len(x),2)); S = np.ones(len(x))
    baselines = {k:models[k].baseline.set_index('month').dH0 for k in [1,2]}
    times = np.union1d(baselines[1].index, baselines[2].index)
    pointer = 0
    for j, horizon in enumerate(h):
        while pointer < len(times) and times[pointer] <= horizon:
            t = times[pointer]
            dh = np.array([baselines[k].get(t,0.) for k in [1,2]])
            individual = rate*dh*(starts<t)[:,None]
            total = individual.sum(axis=1)
            share = np.divide(individual, total[:,None], out=np.zeros_like(individual), where=total[:,None]>0)
            q = -np.expm1(-total)
            F += (S*q)[:,None]*share
            S *= np.exp(-total)
            pointer += 1
        out[:,j,:2], out[:,j,2] = F, S
    assert np.isfinite(out).all() and out.min() >= -1e-12 and out.max() <= 1+1e-12
    assert (np.diff(out[:,:,:2],axis=1) >= -1e-12).all()
    np.testing.assert_allclose(out.sum(axis=2),1.,atol=1e-10)
    return out
