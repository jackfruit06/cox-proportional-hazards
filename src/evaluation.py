"""Entry-aware diagnostics and censoring/competing-risk validation.

Absolute-risk validation uses a common origin subgroup (L=0), or a stated
landmark. This deliberately avoids treating a late entrant's unobserved history
as an observed event-free baseline history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.special import expit, logit
from scipy.stats import chi2, beta as beta_distribution
from statsmodels.stats.multitest import multipletests

from cox_model import risk_moments, predict_cif


FUNCTIONAL_SPECS = {
    'A_linear': {1:(), 2:()},
    'B_FICO_RCS': {1:('classic_fico_score',), 2:('classic_fico_score',)},
    'C_targeted_RCS': {1:('classic_fico_score',),
                       2:('classic_fico_score','original_ltv','original_upb')},
}


def specification_encoder(train, specification):
    from preprocessing import Encoder
    spec = FUNCTIONAL_SPECS[specification]
    columns = list(dict.fromkeys([c for k in [1,2] for c in spec[k]]))
    encoder = Encoder().fit(train)
    if columns:
        encoder = encoder.with_rcs(train, columns)
    by_cause = {k:[c for c in encoder.columns if not c.endswith('__rcs')
                   or c[:-5] in spec[k]] for k in [1,2]}
    return encoder, by_cause


def internal_folds(train, nfolds=3, seed=20262001):
    """Fixed training-only, outcome-stratified partitions shared by candidates."""
    rng = np.random.default_rng(seed)
    groups = [[] for _ in range(nfolds)]
    for _, data in train.groupby('J', sort=True):
        ids = data.index.to_numpy().copy()
        rng.shuffle(ids)
        for fold, part in enumerate(np.array_split(ids,nfolds)):
            groups[fold].extend(part.tolist())
    return groups


def development_losses(models, encoder, training, validation, horizons=(36,60,120)):
    """OOF loss only on common-origin validation; never receives legacy test."""
    tr, va = landmark_sample(training,0), landmark_sample(validation,0)
    curve = censoring_curve(tr)
    probabilities = predict_cif(models,encoder.transform(va),horizons)[:,:,0]
    losses = np.column_stack([w*(y-probabilities[:,j])**2
                for j,h in enumerate(horizons) for y,w in [ipcw_status(va,h,curve)]])
    return pd.DataFrame(losses,index=va.index,columns=horizons), probabilities


def functional_experiments(train, nfolds=3):
    """Compare A/B/C with fold-specific encoding/knots, then choose on train.

    Locked rule: B addresses replicated FICO nonlinearity if each training LR
    p<.01 and OOF loss is not worse by more than one paired fold SE. Add the
    two Cause-2 terms in C only for >0.5% relative OOF improvement AND >one
    paired fold SE. Fold SE is a parsimony heuristic, not a formal coverage CI.
    All fits must converge without warnings. No legacy holdout is accepted.
    """
    from cox_model import fit_models
    from scipy.stats import chi2
    from preprocessing import sha256
    candidates, rows = {}, []
    folds = internal_folds(train,nfolds)
    for name in FUNCTIONAL_SPECS:
        encoder, by_cause = specification_encoder(train,name)
        x = encoder.transform(train)
        models = fit_models(train,x,by_cause)
        for k,m in models.items():
            if m.warnings:
                raise RuntimeError(f'{name} Cause {k} fit warnings: {m.warnings}')
            diag = diagnostics(train,x,m)
            rows.append({'specification':name,'cause':k,'parameters':len(m.beta),
                         'log_likelihood':m.log_likelihood,'AIC':-2*m.log_likelihood+2*len(m.beta),
                         'max_abs_score':m.max_abs_score,'min_covariance_eigenvalue':float(np.linalg.eigvalsh(m.covariance).min()),
                         'PH_global_chi2':float(diag['ph'].iloc[0].chi2),
                         'PH_global_p':float(diag['ph'].iloc[0].p)})
        candidates[name]={'encoder':encoder,'columns_by_cause':by_cause,'models':models,
                          'oof_losses':[], 'fold_models':[], 'fold_scores':[]}
    # Lock fold identity before observing any candidate validation scores.
    for fold, ids in enumerate(folds):
        validation = train.loc[ids]
        training = train.loc[~train.index.isin(ids)]
        for name, item in candidates.items():
            encoder, by_cause = specification_encoder(training,name)
            assert not set(encoder.training_ids)&set(validation.index)
            models = fit_models(training,encoder.transform(training),by_cause)
            if any(m.warnings for m in models.values()):
                raise RuntimeError(f'OOF {name} fold {fold} convergence warning')
            loss, probabilities = development_losses(models,encoder,training,validation)
            item['oof_losses'].append(loss)
            item['fold_scores'].append(float(loss.to_numpy().mean()))
            item['fold_models'].append({'encoder':encoder,'models':models,
                                         'training_ids':training.index.tolist(),
                                         'validation_ids':validation.index.tolist(),
                                         'losses':loss,'predictions':probabilities})
    for item in candidates.values():
        item['oof_losses']=pd.concat(item['oof_losses']).loc[train.index[train.L.eq(0)]]
    table = pd.DataFrame(rows)
    for name,item in candidates.items():
        table.loc[table.specification.eq(name),'OOF_Brier_mean_36_60_120']=item['oof_losses'].to_numpy().mean()
    lr=[]
    for smaller,larger in [('A_linear','B_FICO_RCS'),('B_FICO_RCS','C_targeted_RCS')]:
        for k in [1,2]:
            a,b=candidates[smaller]['models'][k],candidates[larger]['models'][k]
            df=len(b.beta)-len(a.beta)
            if df:
                statistic=2*(b.log_likelihood-a.log_likelihood)
                lr.append({'smaller':smaller,'larger':larger,'cause':k,'df':df,
                           'LR':statistic,'p':float(chi2.sf(statistic,df))})
    comparisons=[]
    for smaller,larger in [('A_linear','B_FICO_RCS'),('B_FICO_RCS','C_targeted_RCS')]:
        a,b=candidates[smaller],candidates[larger]
        differences=np.array(a['fold_scores'])-np.array(b['fold_scores'])
        gain=float(a['oof_losses'].to_numpy().mean()-b['oof_losses'].to_numpy().mean())
        comparisons.append({'smaller':smaller,'larger':larger,'OOF_Brier_gain':gain,
                            'paired_fold_SE':float(differences.std(ddof=1)/np.sqrt(nfolds)),
                            'relative_gain':gain/float(a['oof_losses'].to_numpy().mean())})
    b,c=comparisons
    fico_evidence=all(r['p']<.01 for r in lr if r['larger']=='B_FICO_RCS')
    selected='B_FICO_RCS' if fico_evidence and b['OOF_Brier_gain']>=-b['paired_fold_SE'] else 'A_linear'
    if selected=='B_FICO_RCS' and c['relative_gain']>.005 and c['OOF_Brier_gain']>c['paired_fold_SE']:
        selected='C_targeted_RCS'
    return {'candidates':candidates,'comparison':table,'likelihood_ratios':pd.DataFrame(lr),
            'paired_comparisons':pd.DataFrame(comparisons),'selected':selected,
            'folds':folds,'nfolds':nfolds,'selection_horizons':[36,60,120],
            'selection_rule':functional_experiments.__doc__,
            'training_ids':train.index.tolist(),'legacy_test_used':False}


def time_diagnostics(data,x,model):
    """Score tests for remaining constant coefficients, conditional on TV terms.

    Already time-varying columns are excluded from a redundant PH test. Scaled
    residuals/one-step influence use the actual event-time design and weights.
    """
    from cox_model import time_risk_moments
    x=x[model.static_columns]
    p=len(model.beta)
    tested=[i for i,c in enumerate(model.static_columns) if c not in model.time_effects]
    columns=[model.static_columns[i] for i in tested]
    q=len(tested)
    I=np.zeros((p,p));cross=np.zeros((p,q));extra=np.zeros((q,q));U=np.zeros(q)
    individual_score=np.zeros((len(data),p));integrated=np.zeros(len(data));scaled_rows=[]
    nevents=int(data.J.eq(model.cause).sum())
    for t,event,risk,wr,we,weights,logden,mean,variance in time_risk_moments(data,x,model.beta,model.cause,model.time_effects):
        deaths=int(event.sum());g=np.log(t/36.)
        residual=we-mean
        I+=deaths*variance
        cross+=deaths*variance[:,tested]*g
        extra+=deaths*variance[np.ix_(tested,tested)]*g*g
        U+=g*residual[:,tested].sum(axis=0)
        individual_score[event]+=we-mean
        individual_score[risk]-=(deaths*weights)[:,None]*(wr-mean)
        integrated[risk]+=deaths*weights
        scaled=nevents*residual@model.covariance
        scaled_rows.extend([[t,*row] for row in scaled])
    np.testing.assert_allclose(I,np.linalg.inv(model.covariance),rtol=1e-7,atol=1e-5)
    np.testing.assert_allclose(individual_score.sum(axis=0),0.,atol=1e-4)
    efficient=extra-cross.T@np.linalg.solve(I,cross)
    statistic=float(U@np.linalg.solve(efficient,U))
    rows=[{'variable':'GLOBAL','chi2':statistic,'df':q,'p':float(chi2.sf(statistic,q))}]
    for j,c in enumerate(columns):
        stat=float(U[j]**2/efficient[j,j])
        rows.append({'variable':c,'chi2':stat,'df':1,'p':float(chi2.sf(stat,1))})
    ph=pd.DataFrame(rows);ph['p_holm']=np.nan
    ph.loc[ph.variable.ne('GLOBAL'),'p_holm']=multipletests(ph.loc[ph.variable.ne('GLOBAL'),'p'],method='holm')[1]
    ph['scope']='remaining constant-coefficient terms, conditional on selected time effect'
    dfbeta=individual_score@model.covariance
    residuals=pd.DataFrame({'martingale':data.J.eq(model.cause).to_numpy(float)-integrated,
        'max_abs_standardized_dfbeta':np.max(np.abs(dfbeta)/np.sqrt(np.diag(model.covariance)),axis=1)},index=data.index)
    return {'ph':ph,'scaled_schoenfeld':pd.DataFrame(scaled_rows,columns=['month',*model.columns]),
            'residuals':residuals,'dfbeta':dfbeta}


def time_experiments(train, development):
    """At most one prespecified effect per cause: FICO (1), rate (2).

    Retain only if training LR p<.01/2, HR contrast changes >=25% between
    ages 12 and 60, reference-profile CIF changes >=0.002 at a selected horizon,
    and relative OOF Brier gain >0.5% AND >one paired fold SE. Otherwise retain
    the smaller working specification and disclose remaining non-PH.
    """
    from cox_model import fit_time_varying, hazard_contrasts, reference_rows
    selected=development['candidates'][development['selected']]
    encoder=selected['encoder'];x=encoder.transform(train)
    effects={1:('classic_fico_score',),2:('original_interest_rate',)}
    alternative={k:fit_time_varying(train,x[selected['models'][k].columns],k,effects[k],selected['models'][k]) for k in [1,2]}
    combinations={'static':selected['models'],
                  'TV_FICO_C1':{1:alternative[1],2:selected['models'][2]},
                  'TV_rate_C2':{1:selected['models'][1],2:alternative[2]},
                  'TV_both':alternative}
    cv={name:[] for name in combinations};fold_models=[]
    for fold in selected['fold_models']:
        training,validation=train.loc[fold['training_ids']],train.loc[fold['validation_ids']]
        enc=fold['encoder'];xx=enc.transform(training)
        tv={k:fit_time_varying(training,xx[fold['models'][k].columns],k,effects[k],fold['models'][k]) for k in [1,2]}
        ms={'static':fold['models'],'TV_FICO_C1':{1:tv[1],2:fold['models'][2]},
            'TV_rate_C2':{1:fold['models'][1],2:tv[2]},'TV_both':tv}
        for name,models in ms.items():
            losses,_=development_losses(models,enc,training,validation)
            cv[name].append(float(losses.to_numpy().mean()))
        fold_models.append(tv)
    rows=[]
    profiles=reference_rows(encoder,9)
    profiles['classic_fico_score']=np.repeat(train.classic_fico_score.quantile([.1,.5,.9]).to_numpy(),3)
    profiles['original_interest_rate']=np.tile(train.original_interest_rate.quantile([.1,.5,.9]).to_numpy(),3)
    px=encoder.transform(profiles)
    base=predict_cif(combinations['static'],px,[12,36,60,120])[:,:,0]
    for name,models in combinations.items():
        differences=np.array(cv['static'])-np.array(cv[name])
        delta=float(differences.mean());se=float(differences.std(ddof=1)/np.sqrt(len(differences)))
        pred=predict_cif(models,px,[12,36,60,120])[:,:,0]
        rows.append({'specification':name,'OOF_Brier_mean':float(np.mean(cv[name])),
                     'OOF_gain':delta,'paired_fold_SE':se,'relative_gain':delta/float(np.mean(cv['static'])),
                     'max_profile_CIF_difference':float(np.max(np.abs(pred-base)))})
    evidence=[]
    for k,name in [(1,'TV_FICO_C1'),(2,'TV_rate_C2')]:
        model=alternative[k];variable=effects[k][0]
        lr=2*(model.log_likelihood-selected['models'][k].log_likelihood)
        value=encoder.medians[variable]+encoder.continuous[variable]
        hr12=float(hazard_contrasts(model,encoder,variable,[value],12).HR.iloc[0])
        hr60=float(hazard_contrasts(model,encoder,variable,[value],60).HR.iloc[0])
        row=next(r for r in rows if r['specification']==name)
        material_ratio=max(hr12/hr60,hr60/hr12)
        accept=(chi2.sf(lr,1)<.005 and material_ratio>=1.25
                and row['max_profile_CIF_difference']>=.002
                and row['relative_gain']>.005 and row['OOF_gain']>row['paired_fold_SE'])
        evidence.append({'cause':k,'variable':variable,'training_LR':float(lr),'p':float(chi2.sf(lr,1)),
                         'HR_12':hr12,'HR_60':hr60,'HR_ratio_change':material_ratio,'retain':bool(accept)})
    retain=[r['cause'] for r in evidence if r['retain']]
    choice='TV_both' if len(retain)==2 else 'TV_FICO_C1' if retain==[1] else 'TV_rate_C2' if retain==[2] else 'static'
    return {'models':combinations[choice],'selected':choice,'alternatives':alternative,
            'comparisons':pd.DataFrame(rows),'evidence':pd.DataFrame(evidence),
            'fold_models':fold_models,'fold_scores':cv,'selection_rule':time_experiments.__doc__,
            'legacy_test_used':False}


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
    dynamic = callable(lp)
    score_function = lp if dynamic else None
    lp = None if dynamic else np.asarray(lp,float)
    w = np.ones(len(data)) if counts is None else np.asarray(counts,float)
    if not dynamic:
        order = np.argsort(lp,kind='stable')
        scores = lp[order]
    numerator, denominator = 0.,0.
    for t in np.unique(T[(J==1)&(w>0)]):
        if dynamic:
            lp = np.asarray(score_function(float(t)),float)
            order = np.argsort(lp,kind='stable')
            scores = lp[order]
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
    def objective(b):
        # Elementwise two-parameter algebra avoids thousands of tiny BLAS
        # dispatches in test bootstrap, without changing the objective/score.
        eta=b[0]+b[1]*z
        loss=np.sum(w*(np.logaddexp(0,eta)-y*eta))/w.sum()
        residual=w*(expit(eta)-y)
        grad=np.array([residual.sum(),np.sum(residual*z)])/w.sum()
        return loss,grad
    result = minimize(objective,[0.,1.],jac=True,method='BFGS',options={'gtol':1e-8,'maxiter':200})
    loss, gradient = objective(result.x)
    if not np.isfinite(result.x).all() or np.max(np.abs(gradient)) > 1e-5:
        return np.nan,np.nan,False
    return float(result.x[0]),float(result.x[1]),True


def calibration_offset(p,y,w):
    """Calibration intercept with logit(p) offset and slope FIXED at one."""
    p,y,w=[np.asarray(v,float) for v in [p,y,w]]
    positive=float(w@y);total=float(w.sum())
    if positive<=0 or positive>=total:
        return np.nan
    z=logit(np.clip(p,1e-8,1-1e-8))
    return float(brentq(lambda a:np.sum(w*(expit(z+a)-y)),-60,60))


def cif_auc(p,y,w):
    """IPCW cumulative/dynamic AUC; competing events are known-zero controls."""
    p,y,w=[np.asarray(v,float) for v in [p,y,w]]
    order=np.argsort(p,kind='stable');scores=p[order]
    cases=w*y;controls=w*(1-y)
    denominator=cases.sum()*controls.sum()
    if denominator<=0:return np.nan
    cumulative=np.r_[0.,np.cumsum(controls[order])]
    low=np.searchsorted(scores,p,side='left');high=np.searchsorted(scores,p,side='right')
    return float(np.sum(cases*(cumulative[low]+.5*(cumulative[high]-cumulative[low])))/denominator)


def individual_predictions(models, encoder, cohort, train_ids, horizons=(12,36,60,120)):
    """Rank common-origin L=0 loans; no incomparable cross-entry risk rankings."""
    sample=cohort.loc[cohort.L.eq(0)]
    probabilities=predict_cif(models,encoder.transform(sample),horizons)
    roles=np.where(sample.index.isin(train_ids),'training','reused internal holdout')
    rows=[]
    for j,h in enumerate(horizons):
        table=pd.DataFrame({'loan_identifier':sample.index,'sample_role':roles,
            'prediction_population':'L=0 common FPD origin','start':0,'horizon':h,
            'predicted_CIF_1':probabilities[:,j,0],'predicted_CIF_2':probabilities[:,j,1],
            'survival':probabilities[:,j,2]})
        table['risk_rank_at_horizon']=table.predicted_CIF_1.rank(method='min',ascending=False).astype(int)
        rows.append(table)
    return pd.concat(rows,ignore_index=True)


def landmark_sample(data, start):
    take = data.L.eq(0) if start==0 else (data.L.le(start)&data['T'].gt(start))
    out = data.loc[take].copy()
    out['L'] = 0
    out['T'] = out['T']-start
    assert (out['T']>0).all()
    return out


def validate(models, encoder, train, test, *, start=0, horizons=(12,36,60,120),
             bootstrap=200, seed=20261115, integration='exponential'):
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
    kwargs = {} if integration=='exponential' else {'method':integration}
    predictions = predict_cif(models,encoder.transform(test.loc[sample.index]),hs,start=start,**kwargs)[:,:,0]
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
            joint_intercept=intercept,joint_slope=slope,
            CITL_offset=calibration_offset(p,y,w),cif_auc=cif_auc(p,y,w),
            brier=float(np.mean(w*(y-p)**2)),
            benchmark_predicted=float(benchmark[j]),
            benchmark_brier=float(np.mean(w*(y-benchmark[j])**2))))
        labels=pd.qcut(p,5,labels=False,duplicates='drop')
        for q in np.unique(labels):
            mask=labels==q
            aj=float(aalen_johansen(sample.iloc[np.flatnonzero(mask)],[h-start]).F1.iloc[0])
            groups.append(dict(horizon=int(h),landmark=start,quintile=int(q)+1,n=int(mask.sum()),
                               predicted=float(p[mask].mean()),observed=aj,
                               cause1_events_h=int(y[mask].sum()),unknown_h=int((w[mask]==0).sum())))
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
            draws[-1].update(CITL_offset=calibration_offset(p,y,w),cif_auc=cif_auc(p,y,w))
            # Quintile cut points fixed at original validation predictions; AJ refit in resample.
            labels=pd.qcut(predictions[:,j],5,labels=False,duplicates='drop')
            for q in np.unique(labels):
                mask=labels==q
                val=aalen_johansen(sample.iloc[np.flatnonzero(mask)],[h-start],counts=counts[mask]).F1.iloc[0]
                group_draws.append((b,int(h),int(q)+1,float(val)))
    boot=pd.DataFrame(draws)
    if bootstrap:
        for metric in ['mean_predicted','observed_AJ','calibration_difference','observed_expected',
                       'brier','benchmark_brier','brier_improvement','intercept','slope','CITL_offset','cif_auc']:
            interval=boot.groupby('horizon')[metric].quantile([.025,.975]).unstack()
            metrics[metric+'_low']=metrics.horizon.map(interval[.025])
            metrics[metric+'_high']=metrics.horizon.map(interval[.975])
        metrics['bootstrap_success']=metrics.horizon.map(boot.groupby('horizon').brier.count())
        metrics['calibration_bootstrap_success']=metrics.horizon.map(boot.groupby('horizon').calibration_fit_ok.sum())
        metrics['offset_bootstrap_success']=metrics.horizon.map(boot.groupby('horizon').CITL_offset.count())
    group_table=pd.DataFrame(groups)
    if bootstrap:
        gb=pd.DataFrame(group_draws,columns=['replicate','horizon','quintile','observed'])
        ci=gb.groupby(['horizon','quintile'])['observed'].quantile([.025,.975]).unstack()
        group_table=group_table.join(ci.rename(columns={.025:'observed_low',.975:'observed_high'}),on=['horizon','quintile'])
    else:
        group_table['observed_low']=np.nan;group_table['observed_high']=np.nan
    group_table['CI_method']='test-loan percentile AJ bootstrap' if bootstrap else 'not estimated'
    group_table['bootstrap_degenerate']=False
    for i,row in group_table.iterrows():
        degenerate=bool(bootstrap and np.isfinite(row.observed_low) and row.observed_low==row.observed_high)
        group_table.loc[i,'bootstrap_degenerate']=degenerate
        boundary=row.cause1_events_h==0 or row.cause1_events_h==row.n
        if boundary:
            if row.unknown_h==0:
                events,n=int(row.cause1_events_h),int(row.n)
                group_table.loc[i,'observed_low']=0. if events==0 else beta_distribution.ppf(.025,events,n-events+1)
                group_table.loc[i,'observed_high']=1. if events==n else beta_distribution.ppf(.975,events+1,n-events)
                group_table.loc[i,'CI_method']='exact binomial boundary; all horizon outcomes observed'
            else:
                group_table.loc[i,['observed_low','observed_high']]=np.nan
                group_table.loc[i,'CI_method']='boundary uninformative with censoring; no zero-width CI'
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
    if getattr(model,'time_effects',()):
        return time_diagnostics(data,x,model)
    x=x[model.columns]
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
    x=x[model.columns]
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
