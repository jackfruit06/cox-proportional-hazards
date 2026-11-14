"""P2 cohort/risk-set validation without fitting any Cox model.

Full raw checks are opt-in: COX_FULL_COHORT_TESTS=1. No tables, figures or
models are written by these tests. FullCohortP2.results holds checkpoint facts.
Run this file directly for the complete 37-test no-fit P2 suite, including
P1 regressions and the existing non-fitting method tests.
"""
import hashlib
import inspect
import json
import os
import pickle
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from preprocessing import audit_sources, build_cohort, split_loans, sha256
from cox_model import CoxFit, fit_cause, risk_moments
from evaluation import (aalen_johansen, censoring_curve, concordance, diagnostics,
                        g_at, ipcw_status, landmark_sample, validate)


def reference_terminal(row, priority=2, zb_clock='effective'):
    """Independent scalar candidate ordering, not vectorized build_cohort logic."""
    candidates = []
    zfield = 'zb_effective_month' if zb_clock == 'effective' else 'zb_report_month'
    for field, cause, source in [('event_month', 1, 'NUMERIC_90'),
                                  ('first_ra_month', 2, 'RA'), (zfield, 2, 'ZB')]:
        if pd.notna(row[field]):
            # Cause priority wins first; ZB over RA is only a source-label tie.
            candidates.append((int(row[field]), cause != priority, source == 'RA', cause, source))
    if candidates:
        month, _, _, cause, source = min(candidates)
        return cause, source, month
    return 0, 'CENSOR', int(row['last_month'])


def assert_cohort_policy(case, cohort):
    """Every retained row must agree with the scalar policy and source fields."""
    case.assertTrue(cohort.index.is_unique)
    case.assertFalse(cohort.loan_identifier.duplicated().any())
    case.assertTrue(cohort.index.equals(pd.Index(cohort.loan_identifier, name=cohort.index.name)))
    case.assertTrue(cohort.J.isin([0, 1, 2]).all())
    case.assertEqual(int(sum(cohort.J.eq(k).sum() for k in [0, 1, 2])), len(cohort))
    case.assertTrue(np.isfinite(cohort[['L', 'T', 'terminal_month', 'follow_up']]).all().all())
    case.assertTrue((cohort.L >= 0).all())
    case.assertTrue((cohort['T'] > 0).all())
    case.assertTrue((cohort.L < cohort['T']).all())
    np.testing.assert_array_equal(cohort.L, np.maximum(0, cohort.first_month-cohort.first_payment_date_month))
    np.testing.assert_array_equal(cohort['T'], cohort.terminal_month-cohort.first_payment_date_month+1)
    np.testing.assert_array_equal(cohort.follow_up, cohort['T']-cohort.L)
    case.assertTrue(cohort.terminal_month.ge(cohort.first_month).all())
    case.assertTrue(cohort.terminal_month.ge(cohort.first_payment_date_month).all())
    # Month-at-a-time input values are compared independently for every loan.
    expected = [reference_terminal(row) for row in cohort.to_dict('records')]
    case.assertEqual(cohort.J.tolist(), [r[0] for r in expected])
    case.assertEqual(cohort.event_source.tolist(), [r[1] for r in expected])
    np.testing.assert_array_equal(cohort.terminal_month, [r[2] for r in expected])
    for row in cohort.to_dict('records'):
        competing = [row[c] for c in ['first_ra_month', 'zb_effective_month'] if pd.notna(row[c])]
        if competing:
            case.assertEqual(row['competing_month'], min(competing))
        else:
            case.assertTrue(pd.isna(row['competing_month']))
        case.assertEqual(row['same_month'], pd.notna(row['event_month']) and bool(competing)
                         and row['event_month'] == min(competing))
        case.assertEqual(row['zb_ra_same_month'], pd.notna(row['first_ra_month'])
                         and pd.notna(row['zb_effective_month'])
                         and row['first_ra_month'] == row['zb_effective_month'])
        for field in ['event_month', 'first_ra_month', 'zb_effective_month']:
            if pd.notna(row[field]):
                case.assertLessEqual(row['terminal_month'], row[field])


def interval_risk_counts(data, grid):
    """Independent interval sweep: enter at L+1 and leave at T+1 (integer months)."""
    end = int(max(np.max(grid), data['T'].max()))+2
    changes = np.bincount(data.L.to_numpy(int)+1, minlength=end)
    changes -= np.bincount(data['T'].to_numpy(int)+1, minlength=end)
    return np.cumsum(changes)[grid]


def reference_aj(data, grid, counts=None):
    weights = np.ones(len(data)) if counts is None else np.asarray(counts, float)
    L, T, J = [data[c].to_numpy() for c in ['L', 'T', 'J']]
    s, f1, f2 = 1., 0., 0.
    rows = []
    for t in grid:
        # Scalar membership rather than the production vector mask.
        at = [i for i in range(len(data)) if L[i] < t <= T[i]]
        risk = sum(weights[i] for i in at)
        d1 = sum(weights[i] for i in at if T[i] == t and J[i] == 1)
        d2 = sum(weights[i] for i in at if T[i] == t and J[i] == 2)
        if d1+d2:
            old = s
            f1 += old*d1/risk
            f2 += old*d2/risk
            s *= 1-(d1+d2)/risk
        rows.append([s, f1, f2])
    return np.array(rows)


class SyntheticRiskSetsP2(unittest.TestCase):
    @staticmethod
    def boundaries():
        return pd.DataFrame({'L':[0, 1, 0, 2, 3, 0, 1, 2],
                             'T':[1, 3, 3, 4, 5, 4, 2, 3],
                             'J':[1, 2, 1, 1, 0, 2, 0, 2]})

    def test_all_terminal_permutations_and_sources(self):
        import itertools
        triples = list(itertools.product([np.nan, 102, 104], repeat=3))
        ids = [f'loan{i}' for i in range(len(triples))]
        orig = pd.DataFrame({'loan_identifier':ids, 'first_payment_date_month':100}, index=ids)
        hist = pd.DataFrame({'loan_identifier':ids, 'first_month':100, 'last_month':110,
                             'event_month':[r[0] for r in triples],
                             'first_ra_month':[r[1] for r in triples],
                             'zb_effective_month':[r[2] for r in triples],
                             'zb_report_month':[r[2] for r in triples]}, index=ids)
        cohort, flow, exclusions = build_cohort(orig, hist)
        self.assertEqual(len(cohort), 27)
        self.assertTrue(exclusions.empty)
        self.assertEqual(flow.iloc[-1].n, 27)
        assert_cohort_policy(self, cohort)
        # Source/date metadata after a winning event must remain available.
        i = ids[triples.index((102, 104, 104))]
        self.assertEqual(cohort.at[i, 'J'], 1)
        self.assertEqual(cohort.at[i, 'first_ra_month'], 104)
        self.assertEqual(cohort.at[i, 'zb_effective_month'], 104)

    def test_missing_performance_and_nonpositive_followup(self):
        ids = ['missing', 'pre_fpd', 'late_ra', 'eligible']
        orig = pd.DataFrame({'loan_identifier':ids, 'first_payment_date_month':100}, index=ids)
        hist = pd.DataFrame({'loan_identifier':ids[1:], 'first_month':[99, 104, 103],
                             'last_month':[110, 110, 110], 'event_month':np.nan,
                             'first_ra_month':[99, 103, 103], 'zb_effective_month':108,
                             'zb_report_month':108}, index=ids[1:])
        cohort, flow, exclusions = build_cohort(orig, hist)
        self.assertEqual(cohort.index.tolist(), ['eligible'])
        self.assertEqual((cohort.L.iloc[0], cohort['T'].iloc[0]), (3, 4))
        self.assertEqual(exclusions.exclusion.tolist(),
                         ['no_performance', 'nonpositive_follow_up', 'terminal_before_observation'])
        self.assertEqual(int(flow.iloc[1:-1].n.sum())+len(cohort), len(orig))
        self.assertTrue(exclusions.loan_identifier.is_unique)

    def test_known_zb_effective_date_can_follow_last_reporting_month(self):
        orig = pd.DataFrame({'loan_identifier':['loan'], 'first_payment_date_month':[100]}, index=['loan'])
        hist = pd.DataFrame({'loan_identifier':['loan'], 'first_month':[100], 'last_month':[105],
                             'event_month':[np.nan], 'first_ra_month':[np.nan],
                             'zb_effective_month':[106], 'zb_report_month':[105]}, index=['loan'])
        cohort = build_cohort(orig, hist)[0]
        assert_cohort_policy(self, cohort)
        row = cohort.iloc[0]
        self.assertEqual((row.J, row.event_source, row.terminal_month, row['T']), (2, 'ZB', 106, 7))
        reporting = build_cohort(orig, hist, zb_clock='report')[0].iloc[0]
        self.assertEqual((reporting.terminal_month, reporting['T']), (105, 6))

    def test_invariants_detect_inconsistent_metadata_and_duplicate_outcomes(self):
        orig = pd.DataFrame({'loan_identifier':['loan'], 'first_payment_date_month':[100]}, index=['loan'])
        hist = pd.DataFrame({'loan_identifier':['loan'], 'first_month':[100], 'last_month':[110],
                             'event_month':[106], 'first_ra_month':[103],
                             'zb_effective_month':[108], 'zb_report_month':[108]}, index=['loan'])
        cohort = build_cohort(orig, hist)[0]
        for column, value in [('event_source', 'ZB'), ('J', 0), ('terminal_month', 108),
                              ('T', 9), ('L', 1), ('competing_month', 108),
                              ('same_month', True), ('zb_ra_same_month', True)]:
            with self.subTest(column=column):
                damaged = cohort.copy()
                damaged.loc['loan', column] = value
                with self.assertRaises(AssertionError):assert_cohort_policy(self, damaged)
        with self.assertRaises(AssertionError):assert_cohort_policy(self, pd.concat([cohort, cohort]))

    def test_cox_risk_membership_entry_exit_and_cause(self):
        d = self.boundaries()
        x = pd.DataFrame(np.eye(len(d)))
        beta = np.linspace(-.2, .2, len(d))
        for cause in [1, 2]:
            for t, event, deaths, denominator, mean, variance in risk_moments(d, x, beta, cause, second=True):
                ids = [i for i, row in d.iterrows() if row.L < t <= row['T']]
                np.testing.assert_array_equal(np.flatnonzero(mean), ids)
                self.assertAlmostEqual(denominator, np.exp(beta[ids]).sum())
                self.assertEqual(deaths, sum(d.loc[ids, 'T'].eq(t) & d.loc[ids, 'J'].eq(cause)))
                self.assertTrue(np.isfinite(variance).all())
                self.assertFalse(event[d.L.eq(t)].any())
                self.assertTrue(set(np.flatnonzero(event)).issubset(ids))

    def test_phreg_adapter_is_strict_without_fitting(self):
        class ConstructorInspected(Exception): pass
        d = self.boundaries()
        x = pd.DataFrame({'x':np.arange(len(d), dtype=float)})
        for cause in [1, 2]:
            with patch('cox_model.PHReg', side_effect=ConstructorInspected) as constructor:
                with self.assertRaises(ConstructorInspected):fit_cause(d, x, cause)
            args, kwargs = constructor.call_args
            np.testing.assert_array_equal(args[0], d['T'])
            np.testing.assert_allclose(kwargs['entry'], d.L+1e-7, rtol=0, atol=0)
            np.testing.assert_array_equal(kwargs['status'], d.J.eq(cause))
            self.assertEqual(kwargs['ties'], 'breslow')

    def test_aj_boundary_membership_and_zero_weight_replicates(self):
        d = self.boundaries()
        grid = np.arange(0, 7)
        for weights in [None, np.array([2, 0, 1, 3, 1, 0, 2, 1])]:
            np.testing.assert_allclose(aalen_johansen(d, grid, counts=weights),
                                       reference_aj(d, grid, weights), atol=1e-14)

    def test_reverse_km_entry_and_tied_event_censor(self):
        d = self.boundaries()
        curve = censoring_curve(d)
        g = 1.
        for row in curve.itertuples(index=False):
            t = row.month
            at = [i for i, r in d.iterrows() if r.L < t <= r['T']]
            failures = sum(d.loc[at, 'T'].eq(t) & d.loc[at, 'J'].gt(0))
            censored = sum(d.loc[at, 'T'].eq(t) & d.loc[at, 'J'].eq(0))
            self.assertAlmostEqual(row.G_before, g)
            if censored:g *= 1-censored/(len(at)-failures)
            self.assertAlmostEqual(row.G_after, g)

    def test_concordance_entry_exits_and_replication_weights(self):
        d = self.boundaries()
        scores = np.array([3, 1, 1, 2, 2, 0, 1, 3])
        for weights in [np.ones(len(d)), np.array([2, 1, 0, 1, 2, 3, 1, 2])]:
            num, den = 0., 0.
            for i, ri in d.iterrows():
                if ri.J != 1:continue
                for j, rj in d.iterrows():
                    if rj.L < ri['T'] < rj['T']:
                        den += weights[i]*weights[j]
                        num += weights[i]*weights[j]*((scores[i] > scores[j])+.5*(scores[i] == scores[j]))
            result = concordance(d, scores, counts=weights)
            self.assertEqual(result['comparable_pairs'], den)
            self.assertAlmostEqual(result['c_index'], num/den)

    def test_ra_removed_after_exit_in_both_cause_risk_sets(self):
        d = pd.DataFrame({'L':[0, 0, 0, 2], 'T':[2, 3, 4, 5], 'J':[2, 1, 2, 0]},
                         index=['RA', 'numeric', 'ZB', 'late'])
        x = pd.DataFrame(np.eye(len(d)), index=d.index)
        for cause in [1, 2]:
            for t, _, _, _, mean, _ in risk_moments(d, x, np.zeros(len(d)), cause):
                self.assertEqual(bool(mean[0] > 0), t <= 2)
        grid = np.arange(1, 7)
        np.testing.assert_array_equal(interval_risk_counts(d, grid), [3, 3, 3, 2, 1, 0])
        np.testing.assert_allclose(aalen_johansen(d, grid), reference_aj(d, grid))

    def test_diagnostics_integrates_hazard_only_on_observed_interval(self):
        d = pd.DataFrame({'L':[0, 0, 1, 1, 2, 2, 0, 0],
                          'T':[1, 1, 3, 3, 4, 4, 5, 5], 'J':[1, 1, 2, 2, 1, 1, 0, 0]})
        x = pd.DataFrame({'x':[-1., 1.]*4})
        beta = np.zeros(1)
        moments = list(risk_moments(d, x, beta, 1, second=True))
        information = sum(deaths*variance for _, _, deaths, _, _, variance in moments)
        baseline = pd.DataFrame({'month':[r[0] for r in moments], 'dH0':[r[2]/r[3] for r in moments]})
        model = CoxFit(1, ['x'], beta, np.linalg.inv(information), baseline, 0., 0., [])
        result = diagnostics(d, x, model)
        expected = d.J.eq(1).to_numpy(float)
        for i, row in d.iterrows():
            expected[i] -= sum(r[2]/r[3] for r in moments if row.L < r[0] <= row['T'])
        np.testing.assert_allclose(result['residuals'].martingale, expected)
        np.testing.assert_allclose(result['residuals'].loc[[2, 3], 'martingale'], 0.)

    def test_landmark_coordinate_change_and_boundary_eligibility(self):
        d = pd.DataFrame({'L':[0, 2, 12, 13, 0, 0], 'T':[12, 15, 16, 18, 30, 12],
                          'J':[1, 2, 0, 1, 0, 2], 'event_source':['NUMERIC_90', 'RA', 'CENSOR', 'NUMERIC_90', 'CENSOR', 'RA']})
        before = d.copy(deep=True)
        primary = landmark_sample(d, 0)
        self.assertEqual(primary.index.tolist(), [0, 4, 5])
        sample = landmark_sample(d, 12)
        self.assertEqual(sample.index.tolist(), [1, 2, 4])
        self.assertEqual(sample['T'].tolist(), [3, 4, 18])
        self.assertTrue(sample.L.eq(0).all())
        self.assertEqual(sample.event_source.tolist(), ['RA', 'CENSOR', 'CENSOR'])
        pd.testing.assert_frame_equal(d, before)

    def test_ipcw_ra_is_known_zero_at_exit_not_loss_of_followup(self):
        d = pd.DataFrame({'L':[0, 0, 0, 0], 'T':[2, 3, 4, 6], 'J':[2, 1, 0, 0]})
        curve = censoring_curve(d)
        y, w = ipcw_status(d, 2, curve)
        np.testing.assert_array_equal(y, [0, 0, 0, 0])
        np.testing.assert_array_equal(w, [1, 1, 1, 1])
        y, w = ipcw_status(d, 4, curve)
        np.testing.assert_array_equal(y, [0, 1, 0, 0])
        np.testing.assert_allclose(w, [1, 1, 0, 2])

    def test_validate_uses_common_origin_or_landmark_before_ipcw(self):
        n = 620
        d = pd.DataFrame({'L':np.tile([0, 0, 12, 13], n//4),
                          'T':np.r_[np.full(80, 24), np.full(n-80, 100)],
                          'J':np.r_[np.ones(40), np.full(40, 2), np.zeros(n-80)]})
        class Identity:
            def transform(self, data):return pd.DataFrame({'x':np.arange(len(data))}, index=data.index)
        for start in [0, 12]:
            def predictions(models, x, horizons, start):
                p = np.linspace(.01, .2, len(x))
                return np.array([p, p*.5, 1-p*1.5]).T[:, None, :]
            with patch('evaluation.predict_cif', side_effect=predictions) as predictor, \
                 patch('evaluation.calibration_coefficients', return_value=(0., 1., True)):
                result = validate({}, Identity(), d, d, start=start, horizons=[36], bootstrap=0)
            expected = d.index[d.L.eq(0) if start == 0 else (d.L.le(start) & d['T'].gt(start))].tolist()
            self.assertEqual(result['sample_ids'], expected)
            self.assertEqual(predictor.call_args.kwargs['start'], start)
            self.assertEqual(predictor.call_args.args[1].index.tolist(), expected)
            self.assertEqual(int(result['metrics'].n.iloc[0]), len(expected))


@unittest.skipUnless(os.environ.get('COX_FULL_COHORT_TESTS') == '1',
                     'Set COX_FULL_COHORT_TESTS=1 for full raw P2 validation')
class FullCohortP2(unittest.TestCase):
    results = {}

    @classmethod
    def setUpClass(cls):
        cp = json.loads((ROOT/'artifacts/revision_checkpoint.json').read_text(encoding='utf-8'))
        # P1 is nested after P2 is recorded; support repeat validation then too.
        cls.p1 = cp.get('stages', {}).get('P1', cp)
        cls.preserved = cls.p1['preserved_baseline_files_sha256']
        cls.checkpoint = cp
        if 'accepted_P1_build_cohort_sha256' in cp:
            assert hashlib.sha256(inspect.getsource(build_cohort).encode()).hexdigest() == cp['accepted_P1_build_cohort_sha256']
        else:
            assert sha256(ROOT/'src/preprocessing.py') == cls.p1['P1_files_sha256']['src/preprocessing.py']
        cls.orig, cls.hist, cls.audit = audit_sources(ROOT)
        cls.cohort, cls.flow, cls.exclusions = build_cohort(cls.orig, cls.hist)
        cls.train, cls.test = split_loans(cls.cohort)
        frozen = ROOT/'artifacts/baseline_bundle.pkl'
        with (frozen if frozen.exists() else ROOT/'artifacts/research_bundle.pkl').open('rb') as f:
            cls.bundle = pickle.load(f)
        cls.grid = np.arange(1, int(cls.cohort['T'].max())+2)
        cls.results = {'raw_rows':{'origination':len(cls.orig), 'performance':cls.audit['performance_rows']},
                       'cohort_n':len(cls.cohort), 'exclusions_n':len(cls.exclusions),
                       'counts':{str(k):int(cls.cohort.J.eq(k).sum()) for k in [0, 1, 2]},
                       'cause2_sources':cls.cohort.loc[cls.cohort.J.eq(2), 'event_source'].value_counts().to_dict(),
                       'late_entry_n':int(cls.cohort.L.gt(0).sum()),
                       'person_months':int(cls.cohort.follow_up.sum()),
                       'train_n':len(cls.train), 'test_n':len(cls.test),
                       'monthly_grid':{'first':int(cls.grid[0]), 'last':int(cls.grid[-1]), 'n':len(cls.grid)},
                       'model_refits':0}

    def test_every_loan_terminal_source_time_and_exclusivity(self):
        assert_cohort_policy(self, self.cohort)
        self.assertEqual(len(self.cohort), self.p1['cohort_n'])
        self.assertEqual(self.results['counts'], self.p1['after_counts'])
        self.assertEqual(self.results['cause2_sources'], self.p1['cause2_sources'])
        self.assertEqual(int(self.flow.iloc[1:-1].n.sum())+len(self.cohort), len(self.orig))
        self.assertTrue(self.exclusions.loan_identifier.is_unique)
        self.assertFalse(set(self.cohort.index) & set(self.exclusions.loan_identifier))
        beyond_report = self.cohort.terminal_month.gt(self.cohort.last_month)
        known_effective_exit = (self.cohort.J.eq(2) & self.cohort.event_source.eq('ZB')
                                & self.cohort.terminal_month.eq(self.cohort.zb_effective_month)
                                & self.cohort.zb_report_month.le(self.cohort.last_month)
                                & self.cohort.zb_effective_month.gt(self.cohort.last_month))
        np.testing.assert_array_equal(beyond_report, known_effective_exit)
        self.results['known_ZB_effective_exits_after_last_report_n'] = int(beyond_report.sum())
        self.results['max_effective_exit_after_last_report_months'] = int(
            (self.cohort.terminal_month-self.cohort.last_month).loc[beyond_report].max())
        self.results['all_loan_policy_checks'] = True

    def test_all_months_intervals_and_no_contribution_after_any_terminal(self):
        d = self.cohort
        L, T = d.L.to_numpy(), d['T'].to_numpy()
        membership = (L[:, None] < self.grid) & (self.grid <= T[:, None])
        np.testing.assert_array_equal(membership.sum(axis=1), d.follow_up)
        np.testing.assert_array_equal(membership.sum(axis=0), interval_risk_counts(d, self.grid))
        self.assertFalse(membership[self.grid[None, :] > T[:, None]].any())
        self.assertFalse(membership[self.grid[None, :] <= L[:, None]].any())
        self.assertEqual(int(membership.sum()), 3338768)
        self.assertEqual(int(membership[:, -1].sum()), 0)
        self.results['risk_membership_cells_checked'] = int(membership.size)
        self.results['risk_membership_violations'] = 0
        self.results['risk_count_min_max'] = [int(membership.sum(axis=0).min()), int(membership.sum(axis=0).max())]

    def test_ra_exit_and_post_numeric90_records(self):
        d = self.cohort
        ra = d.loc[d.first_ra_month.notna()]
        post90 = ra.event_month.notna() & ra.event_month.lt(ra.first_ra_month)
        self.assertEqual(len(ra), 55)
        self.assertEqual(int(post90.sum()), 54)
        self.assertTrue(ra.loc[post90, 'J'].eq(1).all())
        np.testing.assert_array_equal(ra.loc[post90, 'T'], self.bundle['cohort'].loc[ra.index[post90], 'T'])
        self.assertTrue(ra.terminal_month.le(ra.first_ra_month).all())
        case = d.loc['F15Q30019368']
        self.assertEqual((case.J, case.L, case['T'], case.event_source), (2, 0, 19, 'RA'))
        self.assertEqual(case.first_ra_month, case.terminal_month)
        for t, expected in [(0, False), (1, True), (18, True), (19, True), (20, False), (23, False)]:
            self.assertEqual(bool(case.L < t <= case['T']), expected)
        self.results['RA'] = {'loans':len(ra), 'post90_unchanged':54,
                              'pre90_RA_terminal_age':19, 'risk_at_19':True, 'risk_at_20':False,
                              'post_RA_risk_violations':0}

    def test_train_test_membership_and_accepted_p1_fields(self):
        self.assertEqual(self.train.index.tolist(), self.bundle['train_ids'])
        self.assertEqual(self.test.index.tolist(), self.bundle['test_ids'])
        self.assertFalse(set(self.train.index) & set(self.test.index))
        self.assertEqual(set(self.train.index) | set(self.test.index), set(self.cohort.index))
        old = self.bundle['cohort']
        revised = self.cohort
        pd.testing.assert_frame_equal(old.drop(columns=['T', 'follow_up']),
                                      revised[old.columns].drop(columns=['T', 'follow_up']))
        changed = revised.index[revised['T'].ne(old['T'])].tolist()
        self.assertEqual(changed, ['F15Q30019368'])
        self.results['split_IDs_unchanged'] = True
        self.results['train_ID_sha256'] = hashlib.sha256('\n'.join(self.train.index).encode()).hexdigest()
        self.results['test_ID_sha256'] = hashlib.sha256('\n'.join(self.test.index).encode()).hexdigest()

    def test_cox_moments_and_public_phreg_without_optimization(self):
        d = self.cohort
        # Diagnostic probe covariates/parameters, not a candidate research fit.
        x = pd.DataFrame({'marker':np.linspace(-.5, .5, len(d)), 'entry':d.L.to_numpy()/12}, index=d.index)
        beta = np.array([.1, -.05])
        a = x.to_numpy()
        eta = a@beta
        rr = np.exp(eta)
        result = {}
        with patch.object(PHReg, 'fit', side_effect=AssertionError('P2 forbids fitting')):
            for cause in [1, 2]:
                loglike, score, information = 0., np.zeros(2), np.zeros((2, 2))
                times = 0
                for t, event, deaths, denominator, mean, variance in risk_moments(d, x, beta, cause, second=True):
                    ids = np.flatnonzero((d.L.to_numpy() < t) & (t <= d['T'].to_numpy()))
                    self.assertAlmostEqual(denominator, rr[ids].sum(), places=9)
                    np.testing.assert_allclose(mean, np.average(a[ids], weights=rr[ids], axis=0), atol=1e-13)
                    loglike += eta[event].sum()-deaths*np.log(denominator)
                    score += a[event].sum(axis=0)-deaths*mean
                    information += deaths*variance
                    times += 1
                model = PHReg(d['T'].to_numpy(float), a, status=d.J.eq(cause).to_numpy(int),
                              entry=d.L.to_numpy(float)+1e-7, ties='breslow', missing='raise')
                ll_error = float(abs(loglike-model.loglike(beta)))
                score_error = float(np.max(np.abs(score-model.score(beta))))
                info_error = float(np.max(np.abs(information+model.hessian(beta))))
                self.assertLess(ll_error, 1e-7)
                np.testing.assert_allclose(score, model.score(beta), rtol=1e-9, atol=1e-7)
                np.testing.assert_allclose(information, -model.hessian(beta), rtol=1e-9, atol=1e-7)
                result[str(cause)] = {'event_times':times, 'events':int(d.J.eq(cause).sum()),
                                      'loglike_abs_error':ll_error, 'score_max_abs_error':score_error,
                                      'information_max_abs_error':info_error}
        self.results['Cox_fixed_parameter_risk_checks'] = result

    def test_full_aj_against_independent_interval_sweep(self):
        d, grid = self.cohort, self.grid
        risk = interval_risk_counts(d, grid)
        d1 = np.bincount(d.loc[d.J.eq(1), 'T'], minlength=int(grid[-1])+1)[grid]
        d2 = np.bincount(d.loc[d.J.eq(2), 'T'], minlength=int(grid[-1])+1)[grid]
        expected, s, f1, f2 = [], 1., 0., 0.
        for y, n1, n2 in zip(risk, d1, d2):
            if n1+n2:
                self.assertGreaterEqual(y, n1+n2)
                old = s
                f1 += old*n1/y
                f2 += old*n2/y
                s *= 1-(n1+n2)/y
            expected.append([s, f1, f2])
        got = aalen_johansen(d, grid).to_numpy()
        error = float(np.max(np.abs(got-np.array(expected))))
        np.testing.assert_allclose(got, expected, atol=1e-13, rtol=1e-13)
        self.results['AJ_max_abs_error'] = error

    def test_full_reverse_km_and_validation_population(self):
        d = self.cohort
        curve = censoring_curve(d)
        risk = interval_risk_counts(d, curve.month.to_numpy(int))
        g, errors = 1., []
        for y, row in zip(risk, curve.itertuples(index=False)):
            event = int(((d['T'] == row.month) & d.J.gt(0)).sum())
            censored = int(((d['T'] == row.month) & d.J.eq(0)).sum())
            errors.append(abs(g-row.G_before))
            if censored:g *= 1-censored/(y-event)
            errors.append(abs(g-row.G_after))
        self.assertLessEqual(max(errors), 1e-14)
        populations = {}
        for start in [0, 12]:
            training, sample = landmark_sample(self.train, start), landmark_sample(self.test, start)
            expected = self.test.index[self.test.L.eq(0) if start == 0 else
                                       (self.test.L.le(start) & self.test['T'].gt(start))]
            self.assertTrue(sample.index.equals(expected))
            self.assertTrue(training.L.eq(0).all() and sample.L.eq(0).all())
            np.testing.assert_array_equal(sample['T'], self.test.loc[expected, 'T']-start)
            # Only status/weight construction; no predictions, calibration fits or scores.
            G = censoring_curve(training)
            for h in ([12, 36, 60, 120] if start == 0 else [36, 60, 120]):
                y, w = ipcw_status(sample, h-start, G)
                known_competing = sample.J.eq(2) & sample['T'].le(h-start)
                self.assertTrue((y[known_competing] == 0).all())
                self.assertTrue((w[known_competing] > 0).all())
            populations[str(start)] = {'train_n':len(training), 'test_n':len(sample),
                                        'late_test_included_n':int(self.test.loc[expected, 'L'].gt(0).sum())}
        self.results['reverse_KM_max_abs_error'] = float(max(errors))
        self.results['validation_populations'] = populations

    def test_full_concordance_comparable_count_no_prediction_metric(self):
        d = self.test
        # All scores equal: checks comparator membership and tie logic, not performance.
        expected = 0
        for t in np.unique(d.loc[d.J.eq(1), 'T']):
            cases = int((d['T'].eq(t) & d.J.eq(1)).sum())
            controls = int(((d.L < t) & (t < d['T'])).sum())
            expected += cases*controls
        got = concordance(d, np.zeros(len(d)))
        self.assertEqual(got['comparable_pairs'], expected)
        self.assertEqual(got['c_index'], .5)
        self.results['comparable_pairs_test'] = expected
        self.results['concordance_probe'] = 'all scores tied; membership check only, not model evaluation'

    def test_raw_and_frozen_research_outputs_preserved(self):
        if self.checkpoint.get('baseline_preservation_policy'):
            # P3-P6 are authorized to regenerate current outputs. The original
            # fitted baseline is an immutable byte-for-byte archive; original
            # standalone outputs remain in the recorded Git commit. Do not
            # confuse historical P2 output hashes with current primary hashes.
            for rel,digest in self.preserved.items():
                if rel.startswith('data/'):
                    self.assertEqual(sha256(ROOT/rel),digest,rel)
            self.assertEqual(sha256(ROOT/'artifacts/baseline_bundle.pkl'),
                             self.preserved['artifacts/research_bundle.pkl'])
            self.results['preserved_raw_files'] = 3
            self.results['original_baseline_archive_byte_identical'] = True
            return
        for rel, digest in self.preserved.items():
            self.assertEqual(sha256(ROOT/rel), digest, rel)
        self.results['preserved_raw_artifact_output_files'] = len(self.preserved)


class NoFitP2Suite(unittest.TestSuite):
    def run(self, result, debug=False):
        with patch.object(PHReg, 'fit', side_effect=AssertionError('P2 forbids Cox fitting')):
            return super().run(result, debug=debug)


def p2_suite():
    """Accepted P1 + new P2 + legacy no-fit tests; defer the two fitting tests."""
    import test_methods
    loader = unittest.defaultTestLoader
    suite = NoFitP2Suite()
    for cls in [test_methods.BroaderCompetingExitP1, SyntheticRiskSetsP2, FullCohortP2]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    deferred = {'test_ph_score_matches_finite_difference', 'test_validation_bootstrap_path'}
    for name in loader.getTestCaseNames(test_methods.Invariants):
        if name not in deferred:suite.addTest(test_methods.Invariants(name))
    return suite


if __name__ == '__main__':unittest.main(defaultTest='p2_suite', verbosity=2)
