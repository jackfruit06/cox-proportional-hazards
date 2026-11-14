"""Small synthetic invariants; these are not empirical Freddie Mac results."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cox_model import CoxFit, fit_models, predict_cif
from evaluation import (aalen_johansen, censoring_curve, ipcw_status, concordance,
                        diagnostics, landmark_sample, calibration_coefficients, validate)
from preprocessing import build_cohort, month, Encoder, split_loans


def frame(L,T,J):
    return pd.DataFrame({'L':L,'T':T,'J':J})


class Invariants(unittest.TestCase):
    def test_month(self):
        self.assertEqual(month('201601')-month('201512'),1)
        for s in ['201513','201500','20151','XX']:
            with self.assertRaises(ValueError): month(s)

    def test_aj_entry_and_competing(self):
        d=frame([0,0,1],[1,2,2],[1,2,0])
        a=aalen_johansen(d,[1,2])
        # Late entrant is not at risk at t=1; at t=2 its censoring is after events.
        np.testing.assert_allclose(a.to_numpy(),[[.5,.5,0],[.25,.5,.25]])

    def test_reverse_km_ties(self):
        d=frame([0,0,0,0],[1,1,2,3],[1,0,2,0])
        g=censoring_curve(d)
        self.assertAlmostEqual(g.G_after.iloc[0],2/3)
        y,w=ipcw_status(d,2,g)
        np.testing.assert_allclose(y,[1,0,0,0])
        np.testing.assert_allclose(w,[1,0,1.5,1.5])

    def test_brier_competing_is_known_zero(self):
        d=frame([0,0,0],[1,2,5],[1,2,0])
        y,w=ipcw_status(d,3,censoring_curve(d))
        p=np.array([.7,.2,.1])
        self.assertAlmostEqual(np.mean(w*(y-p)**2),(.09+.04+.01)/3)
        np.testing.assert_allclose(w,1.)

    def test_concordance_against_pairs(self):
        d=frame([0,0,1,0,2,0],[1,2,2,3,4,4],[1,2,1,0,1,1])
        lp=np.array([3,1,1,2,2,0]); w=np.array([2,1,3,1,2,1])
        num,den=0.,0.
        for i in range(len(d)):
            if d.J.iloc[i]!=1: continue
            for j in range(len(d)):
                if d.L.iloc[j]<d['T'].iloc[i]<d['T'].iloc[j]:
                    den+=w[i]*w[j]
                    num+=w[i]*w[j]*((lp[i]>lp[j])+.5*(lp[i]==lp[j]))
        got=concordance(d,lp,w)
        self.assertEqual(got['comparable_pairs'],den)
        self.assertAlmostEqual(got['c_index'],num/den)

    def test_cif_joint_hazard_and_conditioning(self):
        b1=pd.DataFrame({'month':[1.,2.],'dH0':[.1,.2]})
        b2=pd.DataFrame({'month':[1.,2.],'dH0':[.3,.6]})
        models={k:CoxFit(k,['x'],np.array([0.]),np.eye(1),b,0,0,[]) for k,b in [(1,b1),(2,b2)]}
        x=pd.DataFrame({'x':[0.,1.]})
        p=predict_cif(models,x,[1,2])
        np.testing.assert_allclose(p[0,:,0],.25*(1-np.exp(-np.array([.4,1.2]))))
        late=predict_cif(models,x,[1,2],start=np.array([0.,1.]))
        self.assertEqual(late[1,0,0],0.)
        self.assertAlmostEqual(late[1,1,0],.25*(1-np.exp(-.8)))

    def test_event_priority_and_clocks(self):
        orig=pd.DataFrame({'loan_identifier':['a','b'],'first_payment_date_month':[100,100]},index=['a','b'])
        h=pd.DataFrame({'loan_identifier':['a','b'],'first_month':[100,103],'last_month':[110,110],
                        'event_month':[105,105],'zb_effective_month':[105,106],
                        'zb_report_month':[104,105],'first_ra_month':[np.nan,np.nan]},index=['a','b'])
        primary,_,_=build_cohort(orig,h)
        self.assertEqual(primary.J.tolist(),[2,1])
        self.assertEqual(primary.L.tolist(),[0,3])
        self.assertEqual(primary['T'].tolist(),[6,6])
        sensitivity,_,_=build_cohort(orig,h,priority=1)
        self.assertEqual(sensitivity.J.tolist(),[1,1])
        reporting,_,_=build_cohort(orig,h,zb_clock='report')
        self.assertEqual(reporting.J.tolist(),[2,2])

    def test_landmark_conditions(self):
        d=frame([0,2,12,13,0],[12,15,16,18,30],[1,2,0,1,0])
        s=landmark_sample(d,12)
        self.assertEqual(s.index.tolist(),[1,2,4])
        self.assertEqual(s['T'].tolist(),[3,4,18])
        self.assertTrue(s.L.eq(0).all())

    def test_train_only_encoder(self):
        rng=np.random.default_rng(1); n=300
        d=pd.DataFrame({c:rng.uniform(1,4,n) for c in Encoder.continuous})
        d['J']=np.tile([0,1,2],100)
        for c in Encoder.categorical: d[c]='P' if c not in ['number_of_units','number_of_borrowers'] else 1
        d.loc[0,'original_dti']=np.nan
        tr,te=split_loans(d)
        enc=Encoder().fit(tr); before=enc.manifest()
        te=te.copy(); te['original_dti']=1e9; te['occupancy_status']='unseen'
        x=enc.transform(te)
        self.assertEqual(before,enc.manifest())
        self.assertFalse(set(enc.training_ids)&set(te.index))
        self.assertEqual(x.columns.tolist(),enc.columns)

    def test_ph_score_matches_finite_difference(self):
        rng=np.random.default_rng(143)
        n=500; x=pd.DataFrame({'x':rng.normal(size=n),'z':rng.normal(size=n)})
        T=rng.integers(2,22,n); L=rng.integers(0,3,n); L=np.minimum(L,T-1)
        J=rng.choice([0,1,2],n,p=[.3,.3,.4]); d=frame(L,T,J)
        m=fit_models(d,x)[1]; diag=diagnostics(d,x,m)
        ts=np.unique(T[J==1]); weights=np.array([((T==t)&(J==1)).sum() for t in ts])
        g=np.log(ts)-np.average(np.log(ts),weights=weights)
        a=x.to_numpy()
        def likelihood(v):
            value=0.
            for t,gt in zip(ts,g):
                b=v[:2]+gt*v[2:]; eta=a@b
                event=(T==t)&(J==1); risk=(L<t)&(T>=t)
                value+=eta[event].sum()-event.sum()*np.log(np.exp(eta[risk]).sum())
            return value
        v=np.r_[m.beta,0.,0.]; eps=2e-4; basis=np.eye(4)*eps
        gradient=np.array([(likelihood(v+e)-likelihood(v-e))/(2*eps) for e in basis])
        H=np.array([[(likelihood(v+ei+ej)-likelihood(v+ei-ej)-likelihood(v-ei+ej)+likelihood(v-ei-ej))/(4*eps**2)
                     for ej in basis] for ei in basis])
        I=-H; effective=I[2:,2:]-I[2:,:2]@np.linalg.solve(I[:2,:2],I[:2,2:])
        stat=gradient[2:]@np.linalg.solve(effective,gradient[2:])
        self.assertAlmostEqual(stat,diag['ph'].iloc[0].chi2,places=4)

    def test_validation_bootstrap_path(self):
        rng=np.random.default_rng(55); n=800
        d=frame(np.zeros(n),rng.integers(2,100,n),rng.choice([0,1,2],n))
        d['x']=rng.normal(size=n)
        class Identity:
            def transform(self,data): return data[['x']]
        tr,te=d.iloc[:600],d.iloc[600:]
        models=fit_models(tr,tr[['x']])
        v=validate(models,Identity(),tr,te,horizons=[36],bootstrap=2)
        self.assertEqual(v['metrics'].bootstrap_success.iloc[0],2)
        self.assertEqual(len(v['calibration']),5)
        self.assertTrue(v['calibration'].observed_low.notna().all())


class BroaderCompetingExitP1(unittest.TestCase):
    """P1 event-policy regressions; full P2 risk-set validation remains gated."""

    @staticmethod
    def cohort(*, numeric=np.nan, ra=np.nan, zb=np.nan, report=None, first=100,
               last=112, priority=2, zb_clock='effective'):
        orig=pd.DataFrame({'loan_identifier':['loan'], 'first_payment_date_month':[100]}, index=['loan'])
        hist=pd.DataFrame({'loan_identifier':['loan'], 'first_month':[first], 'last_month':[last],
                           'event_month':[numeric], 'first_ra_month':[ra],
                           'zb_effective_month':[zb],
                           'zb_report_month':[zb if report is None else report]}, index=['loan'])
        return build_cohort(orig,hist,priority=priority,zb_clock=zb_clock)

    def test_earliest_exit_and_source(self):
        scenarios=[
            ('numeric_before_ra', dict(numeric=103,ra=105,zb=109), 1,'NUMERIC_90',103),
            ('ra_before_numeric', dict(numeric=106,ra=103,zb=109), 2,'RA',103),
            ('ra_without_numeric_or_zb', dict(ra=103), 2,'RA',103),
            ('zb_before_ra_and_numeric', dict(numeric=106,ra=105,zb=103), 2,'ZB',103),
            ('ra_and_zb_same_month', dict(ra=103,zb=103), 2,'ZB',103),
            ('no_terminal_event', dict(), 0,'CENSOR',112),
        ]
        for name,kwargs,cause,source,terminal in scenarios:
            with self.subTest(name=name):
                cohort,flow,excluded=self.cohort(**kwargs)
                self.assertEqual(len(cohort),1)
                self.assertTrue(excluded.empty)
                row=cohort.iloc[0]
                self.assertEqual((row.J,row.event_source,row.terminal_month),(cause,source,terminal))
                self.assertEqual((row.L,row['T'],row.follow_up),(0,terminal-99,terminal-99))
                self.assertEqual(int(flow.iloc[-1].n),1)
                self.assertEqual(bool(row.zb_ra_same_month),name=='ra_and_zb_same_month')

    def test_same_month_numeric_priority_and_source_tie(self):
        for priority,cause,source in [(2,2,'RA'),(1,1,'NUMERIC_90')]:
            row=self.cohort(numeric=103,ra=103,zb=109,priority=priority)[0].iloc[0]
            self.assertEqual((row.J,row.event_source,row.same_month),(cause,source,True))
        row=self.cohort(numeric=103,ra=103,zb=103)[0].iloc[0]
        self.assertEqual((row.J,row.event_source,row.zb_ra_same_month),(2,'ZB',True))

    def test_zb_clock_does_not_shift_ra_clock(self):
        effective=self.cohort(ra=105,zb=106,report=104)[0].iloc[0]
        reporting=self.cohort(ra=105,zb=106,report=104,zb_clock='report')[0].iloc[0]
        self.assertEqual((effective.event_source,effective['T']),('RA',6))
        self.assertEqual((reporting.event_source,reporting['T']),('ZB',5))

    def test_entry_and_exclusions_use_selected_ra_terminal(self):
        row=self.cohort(ra=105,zb=109,first=103)[0].iloc[0]
        self.assertEqual((row.L,row['T'],row.follow_up),(3,6,3))
        kept,_,excluded=self.cohort(ra=102,zb=109,first=103)
        self.assertTrue(kept.empty)
        self.assertEqual(excluded.exclusion.tolist(),['terminal_before_observation'])
        kept,_,excluded=self.cohort(ra=99,zb=109)
        self.assertTrue(kept.empty)
        self.assertEqual(excluded.exclusion.tolist(),['terminal_before_observation'])

    def test_invalid_policy_is_not_silently_accepted(self):
        for kwargs in [dict(priority=0),dict(zb_clock='typo')]:
            with self.assertRaises(ValueError):self.cohort(**kwargs)


if __name__=='__main__': unittest.main()
