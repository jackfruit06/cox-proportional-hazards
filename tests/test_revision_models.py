"""Regression checks for train-only nonlinear and time-dependent revisions."""
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from preprocessing import Encoder, rcs_component
from cox_model import (CoxFit, predict_cif, model_design, time_likelihood,
                       fit_time_varying, fit_cause, hazard_contrasts)
from evaluation import (internal_folds, specification_encoder, cif_auc,
                         calibration_offset, calibration_coefficients, validate, individual_predictions)
from evaluation import diagnostics


def training_fixture(n=300):
    rng=np.random.default_rng(981)
    d=pd.DataFrame({c:rng.uniform(10,100,n) for c in Encoder.continuous})
    d['classic_fico_score']=rng.uniform(550,850,n)
    d['J']=np.tile([0,1,2],n//3)
    d['L']=0
    d['T']=rng.integers(1,130,n)
    for c in Encoder.categorical:d[c]=1 if c.startswith('number_of') else 'P'
    return d


class FunctionalForms(unittest.TestCase):
    def test_rcs_agrees_with_natural_cubic_interpolant(self):
        knots=[600.,730.,800.]
        grid=np.linspace(knots[0],knots[-1],101)
        reference=CubicSpline(knots,rcs_component(knots,knots),bc_type='natural')
        np.testing.assert_allclose(rcs_component(grid,knots),reference(grid),atol=1e-10)
        for grid in [np.array([400.,450.,500.]),np.array([850.,900.,950.])]:
            np.testing.assert_allclose(np.diff(rcs_component(grid,knots),n=2),0.,atol=1e-10)
        with self.assertRaises(ValueError):rcs_component([1.],[1.,1.,2.])

    def test_train_only_knots_centering_rank_and_transform_immutability(self):
        train=training_fixture()
        encoder=Encoder().fit(train).with_rcs(train,['classic_fico_score','original_upb'])
        np.testing.assert_allclose(encoder.spline_knots['classic_fico_score'],train.classic_fico_score.quantile([.1,.5,.9]))
        x=encoder.transform(train)
        self.assertEqual(np.linalg.matrix_rank(x),len(encoder.columns))
        before=encoder.manifest()
        test=train.iloc[:20].copy()
        test['classic_fico_score']=1e6
        encoder.transform(test)
        self.assertEqual(encoder.manifest(),before)
        with self.assertRaises(ValueError):encoder.with_rcs(train.iloc[:-1],['classic_fico_score'])
        reference=train.iloc[:1].copy()
        for c in encoder.continuous:reference[c]=encoder.medians[c]
        self.assertEqual(encoder.transform(reference).classic_fico_score__rcs.iloc[0],0.)

    def test_targeted_spline_columns_differ_by_cause(self):
        train=training_fixture()
        enc,columns=specification_encoder(train,'C_targeted_RCS')
        self.assertIn('classic_fico_score__rcs',columns[1])
        self.assertNotIn('original_ltv__rcs',columns[1])
        self.assertIn('original_ltv__rcs',columns[2])
        self.assertIn('original_upb__rcs',columns[2])
        self.assertEqual(set(columns[1])|set(columns[2]),set(enc.columns))

    def test_shared_internal_folds_are_exhaustive_and_disjoint(self):
        train=training_fixture()
        folds=internal_folds(train)
        self.assertEqual(folds,internal_folds(train))
        flat=[i for fold in folds for i in fold]
        self.assertEqual(set(flat),set(train.index))
        self.assertEqual(len(set(flat)),len(flat))
        for fold in folds:self.assertEqual(train.loc[fold,'J'].value_counts().nunique(),1)

    def test_cif_uses_actual_model_column_subset(self):
        base=pd.DataFrame({'month':[1.,2.],'dH0':[.1,.2]})
        models={1:CoxFit(1,['x'],np.array([.2]),np.eye(1),base,0.,0.,[]),
                2:CoxFit(2,['x','z'],np.array([-.1,.3]),np.eye(2),base,0.,0.,[])}
        x=pd.DataFrame({'z':[1.,2.],'irrelevant':[1e9,-1e9],'x':[0.,1.]})
        result=predict_cif(models,x,[1,2])
        np.testing.assert_allclose(result.sum(axis=2),1.)
        altered=x.copy();altered['irrelevant']=0.
        np.testing.assert_allclose(result,predict_cif(models,altered,[1,2]))


class TimeDependentModels(unittest.TestCase):
    def test_analytic_time_score_and_hessian_against_finite_difference(self):
        rng=np.random.default_rng(1431);n=120
        x=pd.DataFrame({'x':rng.normal(size=n),'z':rng.normal(size=n)})
        d=pd.DataFrame({'L':rng.integers(0,3,n),'T':rng.integers(3,45,n),'J':rng.choice([0,1,2],n)})
        beta=np.array([.1,-.15,.2]);eps=1e-4;basis=np.eye(3)*eps
        def ll(v):return time_likelihood(d,x,v,1,('x',),False)[0]
        value,score,information=time_likelihood(d,x,beta,1,('x',))
        numerical_score=np.array([(ll(beta+e)-ll(beta-e))/(2*eps) for e in basis])
        numerical_hessian=np.array([[(ll(beta+ei+ej)-ll(beta+ei-ej)-ll(beta-ei+ej)+ll(beta-ei-ej))/(4*eps**2)
                                      for ej in basis] for ei in basis])
        np.testing.assert_allclose(score,numerical_score,rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(information,-numerical_hessian,rtol=1e-5,atol=1e-5)
        self.assertTrue(np.isfinite(value))

    def test_time_fit_convergence_and_static_limit(self):
        rng=np.random.default_rng(354);n=500
        x=pd.DataFrame({'x':rng.normal(size=n),'z':rng.normal(size=n)})
        d=pd.DataFrame({'L':rng.integers(0,3,n),'T':rng.integers(3,75,n),'J':rng.choice([0,1,2],n)})
        static=fit_cause(d,x,1)
        ll,_,_=time_likelihood(d,x,np.r_[static.beta,0.],1,('x',))
        self.assertAlmostEqual(ll,static.log_likelihood,places=8)
        model=fit_time_varying(d,x,1,('x',),static)
        self.assertLess(model.max_abs_score,1e-4)
        self.assertGreaterEqual(model.log_likelihood,static.log_likelihood-1e-8)
        self.assertTrue((np.linalg.eigvalsh(model.covariance)>0).all())
        diagnostic=diagnostics(d,x,model)
        self.assertEqual(int(diagnostic['ph'].iloc[0].df),1)
        self.assertNotIn('x',diagnostic['ph'].variable.tolist())
        self.assertTrue(np.isfinite(diagnostic['residuals'].to_numpy()).all())

    def test_time_specific_cif_and_direct_sensitivity_mass_balance(self):
        baseline=pd.DataFrame({'month':[12.,60.],'dH0':[.1,.2]})
        model=CoxFit(1,['x','x__log_time'],np.array([.2,.3]),np.eye(2),baseline,0,0,[],('x',))
        other=CoxFit(2,['x'],np.array([0.]),np.eye(1),baseline,0,0,[])
        x=pd.DataFrame({'x':[1.,-.5]})
        a=model_design(model,x,12)
        np.testing.assert_allclose(a[:,1],x.x*np.log(12/36))
        for method in ['exponential','direct']:
            p=predict_cif({1:model,2:other},x,[12,60],method=method)
            np.testing.assert_allclose(p.sum(axis=2),1.)
            self.assertTrue((np.diff(p[:,:,:2],axis=1)>=0).all())
        late=predict_cif({1:model,2:other},x,[12,60],start=12)
        np.testing.assert_allclose(late[:,0,:],[ [0,0,1],[0,0,1] ])
        with self.assertRaises(ValueError):predict_cif({1:model,2:other},x,[12,60],start=13)
        bad=CoxFit(2,['x'],np.zeros(1),np.eye(1),baseline.assign(dH0=2.),0,0,[])
        with self.assertRaises(ValueError):predict_cif({1:model,2:bad},x,[12,60],method='direct')


class RevisedValidation(unittest.TestCase):
    def test_elementwise_calibration_matches_matrix_objective(self):
        from scipy.optimize import minimize
        from scipy.special import expit,logit
        rng=np.random.default_rng(329)
        p=rng.uniform(.01,.8,700);z=logit(p);w=rng.uniform(.5,2.,len(p))
        y=rng.binomial(1,expit(.2+1.2*z))
        x=np.column_stack([np.ones(len(z)),z])
        def reference(b):
            eta=x@b
            return np.sum(w*(np.logaddexp(0,eta)-y*eta))/w.sum(),x.T@(w*(expit(eta)-y))/w.sum()
        fitted=minimize(reference,[0.,1.],jac=True,method='BFGS',options={'gtol':1e-8,'maxiter':200})
        a,b,ok=calibration_coefficients(p,y,w)
        self.assertTrue(ok)
        np.testing.assert_allclose([a,b],fitted.x,atol=1e-7,rtol=1e-7)

    def test_cif_auc_against_weighted_case_control_pairs(self):
        p=np.array([.1,.2,.2,.7,.9]);y=np.array([0,1,0,1,0]);w=np.array([1.,2.,3.,1.,0.])
        numerator=denominator=0.
        for i in np.flatnonzero(y):
            for j in np.flatnonzero(1-y):
                denominator+=w[i]*w[j]
                numerator+=w[i]*w[j]*((p[i]>p[j])+.5*(p[i]==p[j]))
        self.assertAlmostEqual(cif_auc(p,y,w),numerator/denominator)

    def test_offset_calibration_score_and_boundary(self):
        from scipy.special import expit,logit
        p=np.array([.1,.2,.4,.8]);y=np.array([0,0,1,1]);w=np.array([1.,2.,2.,1.])
        offset=calibration_offset(p,y,w)
        self.assertAlmostEqual(np.sum(w*(expit(logit(p)+offset)-y)),0.,places=9)
        self.assertTrue(np.isnan(calibration_offset(p,np.zeros(4),w)))

    def test_zero_event_intervals_fully_observed_and_censoring(self):
        n=600
        train=pd.DataFrame({'L':np.zeros(n),'T':np.r_[np.full(60,5),np.full(n-60,50)],
                            'J':np.r_[np.ones(60),np.zeros(n-60)]})
        class Identity:
            def transform(self,data):return pd.DataFrame({'x':np.arange(len(data))},index=data.index)
        def predictions(models,x,horizons,start):
            p=np.linspace(.001,.02,len(x))
            return np.array([p,np.zeros(len(x)),1-p]).T[:,None,:]
        for unknown in [False,True]:
            test=pd.DataFrame({'L':np.zeros(n),'T':np.full(n,50),'J':np.zeros(n)})
            if unknown:test.loc[test.index%10==0,'T']=4
            with patch('evaluation.predict_cif',side_effect=predictions), \
                 patch('evaluation.calibration_coefficients',return_value=(np.nan,np.nan,False)):
                result=validate({},Identity(),train,test,horizons=[12],bootstrap=2)
            groups=result['calibration']
            if unknown:
                self.assertTrue(groups.observed_low.isna().all())
                self.assertTrue(groups.CI_method.str.contains('uninformative').all())
            else:
                self.assertTrue(groups.observed_low.eq(0).all())
                self.assertTrue(groups.observed_high.gt(0).all())
                np.testing.assert_allclose(groups.observed_high,1-.025**(1/groups.n))
                self.assertTrue(groups.CI_method.str.contains('exact binomial').all())

    def test_individual_predictions_are_l0_ranked_and_joint_probabilities(self):
        train=training_fixture()
        train.loc[:9,'L']=1
        encoder=Encoder().fit(train)
        baseline=pd.DataFrame({'month':[1.,60.,120.],'dH0':[.01,.02,.03]})
        models={k:CoxFit(k,encoder.columns,np.zeros(len(encoder.columns)),np.eye(len(encoder.columns)),baseline,0.,0.,[]) for k in [1,2]}
        result=individual_predictions(models,encoder,train,train.index[:200])
        self.assertEqual(len(result),290*4)
        self.assertFalse(set(result.loan_identifier)&set(range(10)))
        np.testing.assert_allclose(result[['predicted_CIF_1','predicted_CIF_2','survival']].sum(axis=1),1.)
        self.assertTrue(result.risk_rank_at_horizon.eq(1).all())


if __name__=='__main__':unittest.main()
