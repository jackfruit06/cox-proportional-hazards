"""Entry-aware diagnostics and censoring/competing-risk validation.

Absolute-risk validation uses a common origin subgroup (L=0), or a stated
landmark. This deliberately avoids treating a late entrant's unobserved history
as an observed event-free baseline history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

from cox_model import risk_moments, predict_cif


def aalen_johansen(data, horizons, counts=None):
    L,T,J = [data[c].to_numpy() for c in ['L','T','J']]
    w = np.ones(len(data)) if counts is None else np.asarray(counts,float)
    s, f1, f2, history = 1., 0., 0., [(0.,1.,0.,0.)]
    for t in np.unique(T[(J>0)&(w>0)]):
        risk = w[(L<t)&(T>=t)].sum()
        d1 = w[(T==t)&(J==1)].sum(); d2 = w[(T==t)&(J==2)].sum()
        if risk <= 0:
            raise ValueError('Empty risk set at event time')
        f1 += s*d1/risk; f2 += s*d2/risk
        s *= 1-(d1+d2)/risk
        history.append((t,s,f1,f2))
    a = np.asarray(history)
    indices = np.searchsorted(a[:,0], horizons, side='right')-1
    result = pd.DataFrame(a[indices,1:], columns=['S','F1','F2'], index=horizons)
    np.testing.assert_allclose(result.sum(axis=1),1.,atol=1e-10)
    assert result.min().min() >= -1e-10
    return result


def censoring_curve(train):
    """Reverse KM; tied failures precede censoring, both causes count as failures."""
    L,T,J = [train[c].to_numpy() for c in ['L','T','J']]
    g, rows = 1., []
    for t in np.unique(T):
        risk = ((L<t)&(T>=t)).sum()
        event = ((T==t)&(J>0)).sum()
        censored = ((T==t)&(J==0)).sum()
        before = g
        if censored:
            assert risk-event >= censored
            g *= 1-censored/(risk-event)
        rows.append((t,before,g))
    return pd.DataFrame(rows,columns=['month','G_before','G_after'])


def g_at(curve, t, left=False):
    t = np.asarray(t,float)
    ix = np.searchsorted(curve.month.to_numpy(), t, side='left' if left else 'right')-1
    return np.where(ix>=0, curve.G_after.to_numpy()[np.maximum(ix,0)], 1.)


def ipcw_status(data, horizon, curve):
    T,J = data['T'].to_numpy(),data.J.to_numpy()
    event = (T<=horizon)&(J>0)
    at_risk = T>horizon
    y = ((T<=horizon)&(J==1)).astype(float)
    w = np.zeros(len(data))
    if event.any():
        ge = g_at(curve,T[event],left=True)
        if ge.min() <= 0:
            raise ValueError('Censoring positivity violated at event time')
        w[event] = 1/ge
    gh = float(g_at(curve,horizon))
    if gh <= 0:
        raise ValueError('Censoring positivity violated at horizon')
    w[at_risk] = 1/gh
    return y,w


def concordance(data, lp, counts=None):
    """Harrell-type Cause-1 concordance; controls satisfy L_j<t<T_j.

    All same-time exits are excluded from comparable controls. Replicate weights
    permit an exact loan bootstrap without constructing quadratic pair matrices.
    """
    L,T,J = [data[c].to_numpy() for c in ['L','T','J']]
    lp = np.asarray(lp,float)
    w = np.ones(len(data)) if counts is None else np.asarray(counts,float)
    order = np.argsort(lp,kind='stable')
    scores = lp[order]
    numerator, denominator = 0.,0.
    for t in np.unique(T[(J==1)&(w>0)]):
        control = w*((L<t)&(T>t))
        event = (T==t)&(J==1)&(w>0)
        cum = np.r_[0.,np.cumsum(control[order])]
        low = np.searchsorted(scores,lp[event],side='left')
        high = np.searchsorted(scores,lp[event],side='right')
        numerator += np.sum(w[event]*(cum[low]+.5*(cum[high]-cum[low])))
        denominator += w[event].sum()*control.sum()
    return {'c_index':numerator/denominator if denominator else np.nan,
            'comparable_pairs':float(denominator)}


def calibration_coefficients(p,y,w):
    """IPCW weighted logistic intercept/slope; fixed prediction, not refit Cox."""
    z = logit(np.clip(np.asarray(p),1e-8,1-1e-8))
    x = np.column_stack([np.ones(len(z)),z])
    def objective(b):
        eta=x@b
        loss=np.sum(w*(np.logaddexp(0,eta)-y*eta))/w.sum()
        grad=x.T@(w*(expit(eta)-y))/w.sum()
        return loss,grad
    result = minimize(objective,[0.,1.],jac=True,method='BFGS',options={'gtol':1e-8,'maxiter':200})
    loss, gradient = objective(result.x)
    if not np.isfinite(result.x).all() or np.max(np.abs(gradient)) > 1e-5:
        return np.nan,np.nan,False
    return float(result.x[0]),float(result.x[1]),True


def landmark_sample(data, start):
    take = data.L.eq(0) if start==0 else (data.L.le(start)&data['T'].gt(start))
    out = data.loc[take].copy()
    out['L'] = 0
    out['T'] = out['T']-start
    assert (out['T']>0).all()
    return out


def validate(models, encoder, train, test, *, start=0, horizons=(12,36,60,120),
             bootstrap=200, seed=20261115):
    training = landmark_sample(train,start)
    sample = landmark_sample(test,start)
    curve = censoring_curve(training)
    supported, support = [], []
    for h in horizons:
        if h <= start:
            continue
        t = h-start
        risk = int(training['T'].ge(t).sum())
        events = int((training['T'].le(t)&training.J.eq(1)).sum())
        G = float(g_at(curve,t))
        use = risk>=200 and events>=20 and G>.1
        support.append(dict(horizon=h,landmark=start,train_n=len(training),test_n=len(sample),
                            train_at_risk=risk,train_cause1=events,G=G,supported=use))
        if use:
            supported.append(h)
    if not supported:
        raise ValueError('No predeclared horizon supported by training data')
    hs = np.asarray(supported)
    predictions = predict_cif(models,encoder.transform(test.loc[sample.index]),hs,start=start)[:,:,0]
    benchmark = aalen_johansen(training,hs-start).F1.to_numpy()
    statuses = [ipcw_status(sample,h-start,curve) for h in hs]
    observed = aalen_johansen(sample,hs-start).F1.to_numpy()
    records, groups = [], []
    for j,h in enumerate(hs):
        p=predictions[:,j]; y,w=statuses[j]
        intercept,slope,success=calibration_coefficients(p,y,w)
        records.append(dict(horizon=int(h),landmark=start,n=len(sample),
            test_cause1=int((sample['T'].le(h-start)&sample.J.eq(1)).sum()),
            known_status=int((w>0).sum()),censored_before_or_at_h=int((w==0).sum()),
            effective_sample_size=float(w.sum()**2/np.sum(w*w)),
            max_weight=float(w.max()),weight_sum_over_n=float(w.mean()),
            mean_predicted=float(p.mean()),observed_AJ=float(observed[j]),
            calibration_difference=float(p.mean()-observed[j]),
            observed_expected=float(observed[j]/p.mean()),
            intercept=intercept,slope=slope,calibration_fit_ok=success,
            brier=float(np.mean(w*(y-p)**2)),
            benchmark_predicted=float(benchmark[j]),
            benchmark_brier=float(np.mean(w*(y-benchmark[j])**2))))
        labels=pd.qcut(p,5,labels=False,duplicates='drop')
        for q in np.unique(labels):
            mask=labels==q
            aj=float(aalen_johansen(sample.iloc[np.flatnonzero(mask)],[h-start]).F1.iloc[0])
            groups.append(dict(horizon=int(h),landmark=start,quintile=int(q)+1,n=int(mask.sum()),
                               predicted=float(p[mask].mean()),observed=aj))
    metrics=pd.DataFrame(records)
    metrics['brier_improvement']=metrics.benchmark_brier-metrics.brier
    rng=np.random.default_rng(seed)
    draws, group_draws=[],[]
    for b in range(bootstrap):
        indices=rng.integers(0,len(sample),len(sample))
        counts=np.bincount(indices,minlength=len(sample))
        aj=aalen_johansen(sample,hs-start,counts=counts).F1.to_numpy()
        for j,h in enumerate(hs):
            p=predictions[indices,j]; y,w=[v[indices] for v in statuses[j]]
            intercept,slope,ok=calibration_coefficients(p,y,w)
            bs=float(np.mean(w*(y-p)**2))
            bb=float(np.mean(w*(y-benchmark[j])**2))
            draws.append(dict(replicate=b,horizon=int(h),landmark=start,
                mean_predicted=p.mean(),observed_AJ=aj[j],
                calibration_difference=p.mean()-aj[j],observed_expected=aj[j]/p.mean(),
                brier=bs,benchmark_brier=bb,brier_improvement=bb-bs,
                intercept=intercept,slope=slope,calibration_fit_ok=ok))
            # Quintile cut points fixed at original validation predictions; AJ refit in resample.
            labels=pd.qcut(predictions[:,j],5,labels=False,duplicates='drop')
            for q in np.unique(labels):
                mask=labels==q
                val=aalen_johansen(sample.iloc[np.flatnonzero(mask)],[h-start],counts=counts[mask]).F1.iloc[0]
                group_draws.append((b,int(h),int(q)+1,float(val)))
    boot=pd.DataFrame(draws)
    if bootstrap:
        for metric in ['mean_predicted','observed_AJ','calibration_difference','observed_expected',
                       'brier','benchmark_brier','brier_improvement','intercept','slope']:
            interval=boot.groupby('horizon')[metric].quantile([.025,.975]).unstack()
            metrics[metric+'_low']=metrics.horizon.map(interval[.025])
            metrics[metric+'_high']=metrics.horizon.map(interval[.975])
        metrics['bootstrap_success']=metrics.horizon.map(boot.groupby('horizon').brier.count())
        metrics['calibration_bootstrap_success']=metrics.horizon.map(boot.groupby('horizon').calibration_fit_ok.sum())
    group_table=pd.DataFrame(groups)
    if bootstrap:
        gb=pd.DataFrame(group_draws,columns=['replicate','horizon','quintile','observed'])
        ci=gb.groupby(['horizon','quintile'])['observed'].quantile([.025,.975]).unstack()
        group_table=group_table.join(ci.rename(columns={.025:'observed_low',.975:'observed_high'}),on=['horizon','quintile'])
    return dict(metrics=metrics,calibration=group_table,bootstrap=boot,
                predictions=predictions,sample_ids=sample.index.tolist(),support=pd.DataFrame(support),
                censoring=curve)


def concordance_bootstrap(data,lp,bootstrap=200,seed=20261115):
    point=concordance(data,lp)
    rng=np.random.default_rng(seed)
    draws=[]
    for _ in range(bootstrap):
        ix=rng.integers(0,len(data),len(data))
        draws.append(concordance(data,lp,np.bincount(ix,minlength=len(data)))['c_index'])
    point.update(low=float(np.nanquantile(draws,.025)),high=float(np.nanquantile(draws,.975)),
                 bootstrap_success=int(np.isfinite(draws).sum()))
    return point,np.asarray(draws)


def diagnostics(data,x,model):
    """Formal score PH test, entry-correct residuals and one-step influence.

    All moment calculations use the same Breslow partial likelihood as the fit.
    Efficient information removes estimation of nuisance constant coefficients.
    The log-time interaction score is a formal test, not a residual correlation.
    """
    a=np.asarray(x,float); beta=model.beta; p=len(beta)
    moments=list(risk_moments(data,x,beta,model.cause,second=True))
    nevents=sum(r[2] for r in moments)
    logs=np.array([np.log(r[0]) for r in moments]); ds=np.array([r[2] for r in moments])
    g=logs-np.average(logs,weights=ds)
    I00=np.zeros((p,p)); I01=I00.copy(); I11=I00.copy(); U=np.zeros(p)
    residual_rows=[]
    integrated=np.zeros(len(data)); integrated_mean=np.zeros((len(data),p))
    schoenfeld=np.full((len(data),p),np.nan)
    for j,(t,event,d,s0,mean,variance) in enumerate(moments):
        information=d*variance
        I00+=information; I01+=g[j]*information; I11+=g[j]**2*information
        residual=a[event]-mean
        U+=g[j]*residual.sum(axis=0)
        schoenfeld[event]=residual
        scaled=nevents*residual@model.covariance
        for v in scaled:
            residual_rows.append([t,*v])
        risk=(data.L.to_numpy()<t)&(data['T'].to_numpy()>=t)
        dh=d/s0
        integrated[risk]+=dh
        integrated_mean[risk]+=dh*mean
    # Independent equality to fitted observed information guards the tied/entry implementation.
    np.testing.assert_allclose(I00,np.linalg.inv(model.covariance),rtol=1e-7,atol=1e-5)
    efficient=I11-I01.T@np.linalg.solve(I00,I01)
    global_stat=float(U@np.linalg.solve(efficient,U))
    rows=[dict(variable='GLOBAL',chi2=global_stat,df=p,p=float(chi2.sf(global_stat,p)))]
    for j,c in enumerate(x.columns):
        stat=float(U[j]**2/efficient[j,j])
        rows.append(dict(variable=c,chi2=stat,df=1,p=float(chi2.sf(stat,1))))
    observed=data.J.eq(model.cause).to_numpy(float)
    rr=np.exp(a@beta)
    martingale=observed-rr*integrated
    score=np.nan_to_num(schoenfeld)-rr[:,None]*(a*integrated[:,None]-integrated_mean)
    np.testing.assert_allclose(score.sum(axis=0),0.,atol=1e-5)
    dfbeta=score@model.covariance
    influence=np.max(np.abs(dfbeta)/np.sqrt(np.diag(model.covariance)),axis=1)
    residuals=pd.DataFrame({'martingale':martingale,'max_abs_standardized_dfbeta':influence},index=data.index)
    ph = pd.DataFrame(rows)
    ph['p_holm'] = np.nan
    ph.loc[ph.variable.ne('GLOBAL'),'p_holm'] = multipletests(ph.loc[ph.variable.ne('GLOBAL'),'p'],method='holm')[1]
    return dict(ph=ph,scaled_schoenfeld=pd.DataFrame(residual_rows,columns=['month',*x.columns]),
                residuals=residuals,dfbeta=dfbeta)


def functional_form_scores(data,x,model,columns):
    """One-at-a-time quadratic extensions, train-only efficient score tests.

    This is a targeted lack-of-linearity diagnostic. It does not select a model
    or demonstrate that a quadratic form is the correct alternative.
    """
    a=np.asarray(x,float); rr=np.exp(a@model.beta)
    L,T,J=[data[c].to_numpy() for c in ['L','T','J']]
    info=np.linalg.inv(model.covariance)
    out=[]
    for c in columns:
        z=x[c].to_numpy()**2
        U=0.; Izz=0.; Ixz=np.zeros(x.shape[1])
        for t in np.unique(T[J==model.cause]):
            risk=(L<t)&(T>=t); event=(T==t)&(J==model.cause)
            w=rr[risk]; w=w/w.sum(); d=event.sum()
            mz=w@z[risk]; mx=w@a[risk]
            U+=z[event].sum()-d*mz
            Izz+=d*(w@(z[risk]**2)-mz*mz)
            Ixz+=d*((w*z[risk])@a[risk]-mz*mx)
        efficient=Izz-Ixz@np.linalg.solve(info,Ixz)
        stat=U*U/efficient if efficient>1e-10 else np.nan
        out.append(dict(cause=model.cause,variable=c,chi2=stat,p=float(chi2.sf(stat,1))))
    table = pd.DataFrame(out)
    table['p_holm'] = multipletests(table.p,method='holm')[1]
    return table
