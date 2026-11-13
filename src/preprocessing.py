"""Independent raw-data audit, cohort construction and train-only encoding.

Source layout is read from the supplied Release 47 workbook, never a prior project.
"""
from __future__ import annotations

import csv
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

SEED = 20261114
ZB_CODES = {'01', '02', '03', '09', '15', '16', '96'}
NUMERIC = {
    'classic_fico_score': (300, 850, 9999),
    'original_dti': (0, 65, 999),
    'original_cltv': (6, 200, 999),
    'original_ltv': (6, 105, 999),
    'mortgage_insurance_percentage': (0, 55, 999),
    'number_of_units': (1, 4, 99),
    'number_of_borrowers': (1, 2, 99),
    'vantagescore_4': (300, 850, 9999),
    'original_interest_rate': (0, np.inf, None),
    'original_upb': (0, np.inf, None),
    'original_loan_term': (1, np.inf, None),
}
CATEGORIES = {
    'first_time_homebuyer_indicator': {'Y', 'N', '9'},
    'occupancy_status': {'P', 'I', 'S', '9'},
    'channel': {'R', 'B', 'C', 'T', '9'},
    'prepayment_penalty_indicator': {'Y', 'N'},
    'amortization_type': {'FRM', 'ARM'},
    'property_type': {'CO', 'PU', 'MH', 'SF', 'CP', '99'},
    'loan_purpose': {'P', 'C', 'N', 'R', '9'},
    'super_conforming_flag': {'Y', 'N'},
    'special_eligibility_program': {'H', 'F', 'R', ''},
    'harp_indicator': {'Y', 'N'},
    'property_valuation_method': {'1', '2', '3', '4', '7'},
    'interest_only_indicator': {'Y', 'N'},
}


def month(value: str) -> int:
    """Calendar month index; no silent coercion of malformed YYYYMM."""
    if len(value) != 6 or not value.isdigit():
        raise ValueError(f'Invalid YYYYMM: {value!r}')
    y, m = int(value[:4]), int(value[4:])
    if not (1900 <= y <= 2200 and 1 <= m <= 12):
        raise ValueError(f'Invalid YYYYMM: {value!r}')
    return 12*y + m - 1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_schema(root: Path) -> dict:
    book = openpyxl.load_workbook(root/'data/Freddie_Mac_Sample_2015_Data_Dictionary_Release47.xlsx',
                                 read_only=True, data_only=True)
    schema = {name: [r[3] for r in list(book[name].values)[1:]]
              for name in ['Origination', 'Performance']}
    assert len(schema['Origination']) == 31 and len(schema['Performance']) == 35
    book.close()
    return schema


def audit_sources(root: Path):
    """Full raw scan. Structural errors stop processing; sources are never edited."""
    schema = read_schema(root)
    quality = {'files': {}, 'performance_domains': {}, 'fatal': []}
    for p in sorted((root/'data').iterdir()):
        quality['files'][p.name] = {'bytes': p.stat().st_size, 'sha256': sha256(p)}
    path = root/'data/sample_orig_2015.txt'
    rows = []
    with path.open(encoding='utf-8', newline='') as f:
        for line, row in enumerate(csv.reader(f, delimiter='|'), 1):
            if len(row) != len(schema['Origination']):
                raise ValueError(f'Origination line {line}: {len(row)} fields')
            rows.append([v.strip() for v in row])
    orig = pd.DataFrame(rows, columns=schema['Origination']).set_index('loan_identifier', drop=False)
    if orig.index.duplicated().any():
        raise ValueError('Duplicate origination loan identifiers')
    for c in ['first_payment_date', 'maturity_date']:
        orig[c+'_month'] = orig[c].map(month)
    if not orig.index.str.fullmatch(r'[FA]15Q[1-4]\d{7}').all():
        raise ValueError('Unexpected loan identifier / vintage')
    quality['origination_rows'] = len(orig)
    quality['numeric_domains'] = []
    clean = orig.copy()
    for c, (lo, hi, sentinel) in NUMERIC.items():
        v = pd.to_numeric(orig[c].replace('', np.nan), errors='raise')
        missing, special = v.isna(), v.eq(sentinel) if sentinel is not None else v.isna() & False
        hi_allowed = pd.Series(hi, index=v.index)
        lo_allowed = pd.Series(lo, index=v.index)
        if c in ['original_cltv', 'original_ltv']:
            hi_allowed.loc[orig.harp_indicator.eq('Y')] = 998
            lo_allowed.loc[orig.harp_indicator.eq('Y')] = 1
        bad = (~missing & ~special) & ((v < lo_allowed) | (v > hi_allowed))
        quality['numeric_domains'].append({'variable': c, 'blank': int(missing.sum()),
            'sentinel': int(special.sum()), 'outside_domain': int(bad.sum()),
            'outside_values': orig.loc[bad, c].value_counts().to_dict()})
        # Keep anomalies visible. A domain anomaly requires explicit resolution before modeling.
        if bad.any():
            quality['fatal'].append(f'{c}: {int(bad.sum())} outside documented domain')
        clean[c] = v.mask(special)
    quality['categorical_domains'] = {}
    for c, allowed in CATEGORIES.items():
        quality['categorical_domains'][c] = orig[c].value_counts(dropna=False).to_dict()
        bad = ~orig[c].isin(allowed)
        if bad.any():
            quality['fatal'].append(f'{c}: unexpected {orig.loc[bad,c].unique().tolist()}')
    quality['term_date_disagreements'] = int((clean.maturity_date_month-clean.first_payment_date_month+1 != clean.original_loan_term).sum())
    quality['cltv_below_ltv'] = int((clean.original_cltv < clean.original_ltv).sum())
    # Aggregate contiguous loan histories. Each monthly record is examined, including records after termination.
    columns = schema['Performance']
    ix = {c: i for i, c in enumerate(columns)}
    domains = {c: Counter() for c in ['current_loan_delinquency_status','zero_balance_code',
               'modification_flag','payment_deferral_flag','borrower_assistance_plan'] if c in ix}
    histories, seen, nrows = [], set(), 0
    active, h, previous = None, None, None
    for row in csv.reader((root/'data/sample_perf_2015.txt').open(encoding='utf-8', newline=''), delimiter='|'):
        nrows += 1
        if len(row) != len(columns):
            raise ValueError(f'Performance line {nrows}: {len(row)} fields')
        row = [x.strip() for x in row]
        loan = row[0]
        if loan != active:
            if h is not None:
                histories.append(h)
            if loan in seen:
                raise ValueError('Noncontiguous repeated loan block: '+loan)
            seen.add(loan)
            active, previous = loan, None
            h = dict(loan_identifier=loan, n_months=0, first_month=month(row[1]),
                     last_month=month(row[1]), first_upb=float(row[2]), first_zb=row[8],
                     first_status=row[3], event_month=np.nan, zb_report_month=np.nan,
                     zb_effective_month=np.nan, zb_code='', gaps=0, missing_months=0,
                     records_after_zb=0, age_disagreements=0, unknown_status_months=0,
                     first_unknown_month=np.nan, first_ra_month=np.nan)
        t = month(row[1])
        if previous is not None:
            if t <= previous:
                raise ValueError(f'Duplicate or out-of-order performance month: {loan} {row[1]}')
            if t > previous+1:
                h['gaps'] += 1
                h['missing_months'] += t-previous-1
        if np.isfinite(h['zb_report_month']) and t > h['zb_report_month']:
            h['records_after_zb'] += 1
        previous = t
        h['last_month'], h['n_months'] = t, h['n_months']+1
        for c in domains:
            domains[c][row[ix[c]]] += 1
        s, z, zd = row[3], row[8], row[9]
        if s.isdigit() and 0 <= int(s) <= 99:
            if int(s) >= 3 and not np.isfinite(h['event_month']):
                h['event_month'] = t
        elif s in {'XX', ''}:
            h['unknown_status_months'] += 1
            if not np.isfinite(h['first_unknown_month']):
                h['first_unknown_month'] = t
        elif s == 'RA':
            if not np.isfinite(h['first_ra_month']):
                h['first_ra_month'] = t
        else:
            raise ValueError(f'Unknown delinquency code {s!r} at {loan}')
        if z and z not in ZB_CODES:
            raise ValueError(f'Unknown Zero Balance Code {z!r}')
        if bool(z) != bool(zd):
            raise ValueError(f'Zero Balance Code/date incomplete: {loan}')
        if z:
            effective = month(zd)
            if not np.isfinite(h['zb_report_month']):
                h['zb_report_month'], h['zb_effective_month'], h['zb_code'] = t, effective, z
        # Age is only a consistency audit, not the analysis clock (modifications can reset age).
        if loan in orig.index and not row[7]:
            h['age_disagreements'] += int(int(row[4]) != t-orig.at[loan, 'first_payment_date_month']+1)
    if h is not None:
        histories.append(h)
    hist = pd.DataFrame(histories).set_index('loan_identifier', drop=False)
    quality['performance_rows'] = nrows
    quality['performance_loans'] = len(hist)
    quality['performance_domains'] = {c: dict(v) for c,v in domains.items()}
    quality['orig_only'] = sorted(set(orig.index)-set(hist.index))
    quality['perf_only'] = sorted(set(hist.index)-set(orig.index))
    quality['history_summary'] = {
        'first_upb_zero': int(hist.first_upb.eq(0).sum()),
        'first_zb_present': int(hist.first_zb.ne('').sum()),
        'loans_with_gaps': int(hist.gaps.gt(0).sum()),
        'missing_months': int(hist.missing_months.sum()),
        'loans_with_records_after_zb': int(hist.records_after_zb.gt(0).sum()),
        'records_after_zb': int(hist.records_after_zb.sum()),
        'zb_date_mismatch': int((hist.zb_report_month.notna() & hist.zb_report_month.ne(hist.zb_effective_month)).sum()),
        'loans_age_disagreement_unmodified': int(hist.age_disagreements.gt(0).sum()),
        'loans_unknown_status': int(hist.unknown_status_months.gt(0).sum()),
        'loans_ra': int(hist.first_ra_month.notna().sum()),
    }
    if quality['perf_only']:
        quality['fatal'].append('Performance loans absent from origination')
    (root/'artifacts/data_audit.json').write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding='utf-8')
    if quality['fatal']:
        raise ValueError('Unresolved domain errors; see artifacts/data_audit.json: '+str(quality['fatal']))
    return clean, hist, quality


def build_cohort(orig, hist, *, priority=2, zb_clock='effective'):
    """FPD=0; month-end exits delta+1; entry=max(0, first delta).

    Thus first observed month (delta, delta+1] enters strictly after its left edge;
    no event is placed before observation. ZB effective date is the primary clock.
    """
    d = orig.join(hist.drop(columns='loan_identifier'), how='left', validate='one_to_one')
    fpd = d.first_payment_date_month
    z = d['zb_effective_month' if zb_clock == 'effective' else 'zb_report_month']
    e = d.event_month
    tie = e.notna() & z.notna() & e.eq(z)
    cause1 = e.notna() & (z.isna() | (e < z) | (tie & (priority == 1)))
    cause2 = z.notna() & ~cause1
    d['J'] = np.select([cause1, cause2], [1,2], default=0).astype(int)
    terminal = pd.Series(np.select([cause1,cause2], [e,z], default=d.last_month), index=d.index)
    d['L'] = np.maximum(0, d.first_month-fpd)
    d['T'] = terminal-fpd+1
    d['follow_up'] = d['T']-d['L']
    d['same_month'] = tie
    d['exclusion'] = ''
    rules = [
        ('no_performance', d.first_month.isna()),
        ('terminal_before_observation', terminal < d.first_month),
        ('nonpositive_follow_up', d['T'] <= d['L']),
        ('terminal_before_FPD', terminal < fpd),
    ]
    flow = [{'reason':'input', 'n':len(d), 'cause1':int((d.J==1).sum()), 'cause2':int((d.J==2).sum())}]
    for reason, mask in rules:
        take = mask & d.exclusion.eq('')
        d.loc[take, 'exclusion'] = reason
        flow.append({'reason':reason, 'n':int(take.sum()),'cause1':int((take & d.J.eq(1)).sum()),
                     'cause2':int((take & d.J.eq(2)).sum())})
    keep = d.loc[d.exclusion.eq('')].copy()
    keep[['L','T']] = keep[['L','T']].astype(int)
    assert keep.index.is_unique and (keep['T'] > keep['L']).all()
    assert keep.J.isin([0,1,2]).all() and (keep.L >= 0).all()
    flow.append({'reason':'retained','n':len(keep),'cause1':int(keep.J.eq(1).sum()),'cause2':int(keep.J.eq(2).sum())})
    return keep, pd.DataFrame(flow), d.loc[d.exclusion.ne(''), ['loan_identifier','exclusion','J','L','T']]


def split_loans(cohort, test_fraction=.25, seed=SEED):
    """Deterministic loan-level split stratified only on the three final outcomes."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for _, g in cohort.groupby('J', sort=True):
        ids = g.index.to_numpy().copy()
        rng.shuffle(ids)
        n = int(round(test_fraction*len(ids)))
        test.extend(ids[:n]); train.extend(ids[n:])
    assert not set(train) & set(test)
    return cohort.loc[train].copy(), cohort.loc[test].copy()


class Encoder:
    """Simple interpretable scaling, median imputation and one-hot encoding.

    Rare categories (<1% training prevalence) are pooled into the modal category.
    Unseen test levels use that reference too and are counted in transform audit.
    No outcome or performance variable is available to this transformer.
    """
    continuous = {'classic_fico_score':50, 'original_dti':10, 'original_ltv':10,
                  'original_interest_rate':1, 'original_upb':100000,
                  'original_loan_term':120, 'mortgage_insurance_percentage':10}
    categorical = ['occupancy_status','loan_purpose','property_type',
                   'first_time_homebuyer_indicator','number_of_units','number_of_borrowers']

    def fit(self, train):
        self.training_ids = train.index.tolist()
        self.medians, self.indicators, self.levels, self.references = {}, [], {}, {}
        for c in self.continuous:
            self.medians[c] = float(train[c].median())
            missing = train[c].isna()
            # A one-loan missing indicator would give separation in cause-specific fits.
            if (missing.sum() >= 50 and (missing & train.J.eq(1)).sum() >= 5
                    and (missing & train.J.eq(2)).sum() >= 5):
                self.indicators.append(c)
        for c in self.categorical:
            s = self._categorical(train[c], c)
            counts = s.value_counts()
            ref = counts.index[0]
            self.references[c] = ref
            self.levels[c] = [ref]+sorted([v for v in counts.index if counts[v] >= .01*len(train) and v != ref])
        x = self._matrix(train)
        self.columns = x.columns[x.std(ddof=0).gt(1e-10)].tolist()
        self.dropped_constant = x.columns.difference(self.columns).tolist()
        assert np.linalg.matrix_rank(x[self.columns].to_numpy()) == len(self.columns)
        return self

    @staticmethod
    def _categorical(s, name):
        if name in ['number_of_units','number_of_borrowers']:
            return s.map(lambda x: 'Missing' if pd.isna(x) else str(int(x)))
        return s.replace({'9':'Missing','99':'Missing','':'Missing'}).fillna('Missing').astype(str)

    def _matrix(self, data):
        x = pd.DataFrame(index=data.index)
        for c, scale in self.continuous.items():
            x[c] = (data[c].fillna(self.medians[c])-self.medians[c])/scale
        for c in self.indicators:
            x[c+'__missing'] = data[c].isna().astype(float)
        for c in self.categorical:
            s = self._categorical(data[c], c)
            for level in self.levels[c][1:]:
                x[c+'__'+level] = s.eq(level).astype(float)
        if getattr(self, 'quadratic_fico', False):
            x['classic_fico_score__squared'] = x['classic_fico_score']**2
        return x.astype(float)

    def with_fico_quadratic(self):
        other = copy.deepcopy(self)
        other.quadratic_fico = True
        other.columns = [*self.columns, 'classic_fico_score__squared']
        return other

    def transform(self, data):
        x = self._matrix(data).reindex(columns=self.columns)
        assert np.isfinite(x.to_numpy()).all()
        return x

    def manifest(self):
        return dict(medians=self.medians, scales=self.continuous, missing_indicators=self.indicators,
                    levels=self.levels, references=self.references, columns=self.columns,
                    dropped_constant=self.dropped_constant,
                    quadratic_fico=getattr(self, 'quadratic_fico', False),
                    training_id_sha256=hashlib.sha256('\n'.join(map(str,self.training_ids)).encode()).hexdigest())
