"""Execute -> save -> test -> export -> verify the single research notebook.

Run from the repository: python src/reproduce.py
No model selection uses the reused holdout; the notebook owns the analysis.
"""
from __future__ import annotations

import base64
import argparse
import ast
import hashlib
import json
import math
import numbers
import os
import pickle
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:
    os.environ[key] = '1'
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):digest.update(block)
    return digest.hexdigest()


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,bool) or value is None or isinstance(value,str):return value
    if isinstance(value,numbers.Integral):return int(value)
    if isinstance(value,numbers.Real):return float(value) if math.isfinite(value) else None
    if hasattr(value,'tolist'):return clean(value.tolist())
    raise TypeError(type(value).__name__)


def write_json(path,value):
    Path(path).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def markdown_table(records,columns):
    def format_value(value):
        if value is None:return '—'
        if isinstance(value,float):return f'{value:.7g}'
        return str(value).replace('|','/')
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    lines.extend('| '+' | '.join(format_value(row.get(c)) for c in columns)+' |' for row in clean(records))
    return '\n'.join(lines)


def revision_inventory(nb,commit):
    """Record exact source changes relative to the frozen original Git tree."""
    def definitions(text):
        result={}
        def collect(nodes,prefix=''):
            for node in nodes:
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                    name=prefix+node.name
                    result[name]=ast.dump(node,include_attributes=False)
                    if isinstance(node,ast.ClassDef):collect(node.body,name+'.')
        collect(ast.parse(text).body)
        return result
    functions={}
    for path in sorted((ROOT/'src').glob('*.py')):
        rel=path.relative_to(ROOT).as_posix()
        try:
            before=subprocess.check_output(['git','show',commit+':'+rel],cwd=ROOT,stderr=subprocess.DEVNULL).decode('utf-8')
        except (subprocess.CalledProcessError,OSError):before=''
        old,current=definitions(before),definitions(path.read_text(encoding='utf-8'))
        functions[rel]=[name for name,body in current.items() if old.get(name)!=body]
    try:
        original=json.loads(subprocess.check_output(['git','show',commit+':notebooks/Cox_PH_Internal_Report.ipynb'],cwd=ROOT,stderr=subprocess.DEVNULL))
        cells=[i for i,(a,b) in enumerate(zip(original['cells'],nb.cells)) if ''.join(a['source'])!=''.join(b['source'])]
    except (subprocess.CalledProcessError,OSError):cells=None
    return {'modified_functions':functions,'modified_notebook_source_cells_from_original':cells,
            'basis':'original baseline Git commit; output/metadata-only changes excluded from cell list'}


def render_checkpoints(checkpoint,manifest,tests):
    """Update the existing stage report, preserving P1/P2 historical evidence."""
    path=ROOT/'stage_report.md';marker='<!-- REVISION_P3_P6 -->'
    text=path.read_text(encoding='utf-8').split(marker,1)[0].rstrip()
    text=re.sub(r'\*\*Trạng thái hiện tại:.*?\*\*',
                '**Trạng thái hiện tại: P1–P6 hoàn tất theo execution-policy update; final working model có giới hạn được công khai.**',text,count=1)
    if '## Stage: P2' in text:
        start=text.index('## Stage: P2')
        section=text[start:]
        if '### 6. Acceptance status' in section:
            section=section.split('### 6. Acceptance status',1)[0]+'''### 6. Acceptance status

**PASS — Safe to proceed.** Cohort/risk-set checks đạt. Execution-policy update
của researcher thay gate xin nghiệm thu bằng tự chuyển stage khi criteria đạt.
Trạng thái chờ review ban đầu được giữ như lịch sử, không coi policy update là
một lần researcher review từng numerical result.

### 7. Recommendation for P3

Đã tự chuyển P3 theo ủy quyền mới; P1 event policy và train/test membership
được giữ. P3–P6 results được ghi bên dưới. Không commit/push.
'''
            text=text[:start]+section.rstrip()
    p3,p4,p5,p6=[checkpoint['stages'][key] for key in ['P3','P4','P5','P6']]
    chunks=[text,'',marker,'']
    common_files='`src/preprocessing.py` (encoder only; accepted build_cohort unchanged), `src/cox_model.py`, `src/evaluation.py`, `tests/test_revision_models.py`, notebook và docs liên quan.'
    chunks += ['## Stage: P3 — Nonlinear functional-form experiments','',
      '### 1. Work completed','',
      'So sánh A linear, B FICO RCS ở cả hai causes, C thêm LTV/UPB RCS Cause 2; 3-fold original-train-only validation. Mỗi fold học encoder/knots riêng. Cùng samples/folds và horizons 36/60/120; legacy test không tham gia selection.','',
      '### 2. Tests performed','',
      'RCS khớp independent natural-cubic interpolant, linear tails, centering/rank, immutable train-only knots, targeted cause columns và CIF column subsets đều đạt. P6 chạy toàn suite.','',
      '### 3. Research results','',
      markdown_table(p3['comparison'],['specification','cause','parameters','log_likelihood','OOF_Brier_mean_36_60_120','PH_global_p']),'',
      markdown_table(p3['LR'],['smaller','larger','cause','df','LR','p']),'',
      markdown_table(p3['paired_comparisons'],['smaller','larger','OOF_Brier_gain','paired_fold_SE','relative_gain']),'',
      f"Selected: **{p3['selected']}**. Knots 10/50/90% train: FICO 683/760/803; candidate LTV 49/78/95; UPB 90000/200000/396000. A coefficients tái tạo baseline gốc. B cải thiện shape evidence và OOF; C không đạt extra predictive/complexity rule. Shape plots có joint covariance CI trên train 1–99th percentiles.",'',
      '### 4. Issues and limitations','',
      'Remaining PH vẫn bị bác bỏ. Paired fold SE là selection heuristic với 3 folds, không formal CI. Model-based shape CI chưa bao gồm selection uncertainty; không chọn spline chỉ vì p nhỏ.','',
      '### 5. Acceptance status','',f"**{p3['status']}** — chọn bằng rule đã khóa, không test shopping.",'',
      '### 6. Next-stage recommendation','',
      'Đã chuyển P4 tự động. Files/functions: specification_encoder, internal_folds, functional_experiments; '+common_files,'',
      '## Stage: P4 — Targeted non-PH assessment','',
      '### 1. Work completed','',
      'Thử tối đa một time effect/cause: FICO Cause 1, rate Cause 2, beta(t)=beta+gamma log(t/36). Analytic Breslow likelihood/score/information, strict delayed entry, safeguarded Newton fit; predictions dùng actual t. Không dùng exit T làm covariate.','',
      '### 2. Tests performed','',
      'Gradient/Hessian khớp finite differences; gamma=0 khớp static likelihood; covariance/convergence, time-specific prediction và probability conservation đạt.','',
      '### 3. Research results','',
      markdown_table(p4['comparisons'],['specification','OOF_Brier_mean','OOF_gain','paired_fold_SE','relative_gain','max_profile_CIF_difference']),'',
      markdown_table(p4['evidence'],['cause','variable','training_LR','p','HR_12','HR_60','HR_ratio_change','retain']),'',
      f"Selected time structure: **{p4['selected']}**. LR/HR changes có evidence, nhưng OOF loss kém và profile CIF deltas dưới 0.002; không giữ thêm complexity. HR(t)+CI figure lưu actual fitted experiments, không mô tả các candidates là primary.",'',
      '### 4. Issues and limitations','',
      'Working RCS vẫn non-PH: Cause 1 global p≈1.03e-11; Cause 2 p≈3.57e-91. Đây là prediction-focused parsimonious working specification, không xác nhận HR thật hằng. Nếu TV được chọn sau này, diagnostics chỉ test remaining constant terms conditional on TV.','',
      '### 5. Acceptance status','',f"**{p4['status']}** — limitations được công khai, không bypass một failed convergence/probability criterion.",'',
      '### 6. Next-stage recommendation','',
      'Đã khóa model rồi chuyển P5. Functions: time_likelihood, fit_time_varying, model_design, time_diagnostics, time_experiments; '+common_files,'',
      '## Stage: P5 — CIF prediction and final descriptive validation','',
      '### 1. Work completed','',
      'Giữ exponential-generator primary; direct PI sensitivity có guard combined jump<=1. Reused holdout chỉ dùng sau model lock. Thêm CIF C/D AUC, joint-vs-offset calibration, valid boundary CI và individual CIF/ranks.','',
      '### 2. Tests performed','',
      'Weighted AUC khớp brute pairs/ties; offset score equation; zero-event fully observed/censoring branches; L=0 ranking; direct/exponential/time-specific mass balance, monotonicity và h>=start guards đạt.','',
      '### 3. Research results','',
      markdown_table(p5['metrics'],['horizon','n','mean_predicted','observed_AJ','brier','benchmark_brier','cif_auc','slope']),'',
      markdown_table(p5['metrics'],['horizon','joint_intercept','joint_slope','CITL_offset','cif_auc_low','cif_auc_high']),'',
      markdown_table(p5['baseline_final_paired_Brier'],['horizon','baseline_minus_final_Brier','low','high']),'',
      f"Cause-specific C-index={p5['concordance']['c_index']:.6f}, fixed-model test CI [{p5['concordance']['low']:.6f}, {p5['concordance']['high']:.6f}]. CIF AUC là quantity riêng. Individual CSV: {p5['individual_prediction_rows']:,} rows / {p5['individual_prediction_loans']:,} L=0 loans, bốn horizons, F1/F2/S và rank; training/holdout labels rõ.",'',
      'Quintile zero-event ở 12 tháng có 0/2357, exact upper 95%=0.00156385, thay CI [0,0]. Boundary có censoring không dùng binomial uncritically; flag uninformative/NaN CI. Common administrative support dùng C_admin>120, n=11767 test và 0 early censoring; population khác primary. Priority, ZB clock, RA policy cũ, baseline linear và direct PI đều là declared sensitivities, không dùng chọn model.','',
      '### 4. Issues and limitations','',
      'Slopes tại 36/60/120 vẫn >1; mean gần AJ không chứng minh calibration toàn risk range. Paired baseline-final Brier CI chứa 0 ở cả bốn horizons, chưa xác lập predictive superiority trên holdout. Test đã reused, CI conditional on fitted model/G; không training/selection uncertainty hoặc external validity. Model-based HR associations không causal/lifetime/default/loss.','',
      '### 5. Acceptance status','',f"**{p5['status']}** — probability/error/ranking mechanics đạt; statistical limitations không bị che.",'',
      '### 6. Next-stage recommendation','',
      'Đã chuyển P6. Functions: predict_cif, validate, cif_auc, calibration_offset, individual_predictions; '+common_files,'',
      '## Stage: P6 — Reproducibility, export and reporting','',
      '### 1. Work completed','',
      'Fresh full notebook từ raw; lưu notebook rồi export HTML; tests và artifact verification sau save. Một lệnh versioned `python src/reproduce.py`, không cần script/audit directory tạm. Baseline archive byte-identical; original outputs/code còn trong recorded Git commit.','',
      '### 2. Tests performed','',
      f"**{tests['run']}/{tests['run']} tests PASS**, full raw P2 enabled, 0 failures/errors/skips. All 23 code cells executed in order; no error outputs. Embedded PNG hashes khớp standalone figures; expected CSV/PNG inventory không có stale extras. Source/raw/notebook/HTML/artifact hashes được ghi sau export tại artifacts/verification.json.",'',
      '### 3. Research results','',
      f"Final **{manifest['specification']['functional']} / {manifest['specification']['time']}**, parameters 18/18; FICO knots 683/760/803. Cohort 49980, J1/J2/J0=2019/37305/10656, Cause 2 ZB/RA=37304/1; train/test IDs giữ nguyên. {manifest['figures']} figures, {len(manifest['required_tables'])} required CSV tables. HR/basis coefficients ở cox_coefficients.csv; joint FICO HR/CI ở hazard_contrasts.csv.",'',
      'Exact source-function inventory (so với original baseline):','',
      markdown_table([{'file':path,'changed definitions':', '.join(names)} for path,names in p6['revision_inventory']['modified_functions'].items()],['file','changed definitions']),'',
      'Notebook source cells đổi (0-based): '+str(p6['revision_inventory']['modified_notebook_source_cells_from_original'])+'. Saved outputs được fresh-execute lại cho mọi code cell.','',
      '### 4. Issues and limitations','',
      'Full execution/verification không xác nhận independent censoring/truncation, PH, calibration hoặc external generalization. Raw reader ResourceWarning phát hiện ở P2 đã sửa bằng context manager, không đổi accepted build_cohort. Bootstrap-cell timeout đã được xử lý bằng elementwise calibration objective tương đương và regression test; full rerun cuối đạt. Jupyter ACL trong sandbox cần chạy kernel ở môi trường Windows thông thường; không nới lỏng connection security. Upstream openpyxl DeprecationWarning không làm fail tests; không đổi dependencies để che warning.','',
      '### 5. Acceptance status','',f"**{p6['status']}** — P6 technical criteria đạt, research limitations được giữ trong notebook/contract/README.",'',
      '### 6. Next-stage recommendation','',
      'P1–P6 hoàn tất trong scope được ủy quyền. Không commit/push, không sửa raw, không thay endpoint. Muốn external-vintage generalization cần dữ liệu/thiết kế mới; không tự mở rộng task này.','']
    path.write_text('\n'.join(chunks),encoding='utf-8',newline='\n')


def main():
    import nbformat
    import pandas as pd
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel',default='python3',help='Installed Jupyter kernel name')
    args=parser.parse_args()
    cp_path=ROOT/'artifacts/revision_checkpoint.json'
    cp=json.loads(cp_path.read_text(encoding='utf-8'))
    frozen=ROOT/'artifacts/baseline_bundle.pkl'
    if not frozen.exists():
        raw=subprocess.check_output(['git','show',cp['baseline_commit']+':artifacts/research_bundle.pkl'],cwd=ROOT)
        if hashlib.sha256(raw).hexdigest()!=cp['baseline_bundle_sha256']:
            raise RuntimeError('Cannot recover the exact original baseline bundle')
        frozen.write_bytes(raw)
    assert sha(frozen)==cp['baseline_bundle_sha256']
    cp.update(status='IN PROGRESS',current_stage='P6')
    cp.setdefault('stages',{}).setdefault('P6',{}).update(status='IN PROGRESS',
        started_utc=datetime.now(timezone.utc).isoformat())
    write_json(cp_path,cp)
    path=ROOT/'notebooks/Cox_PH_Internal_Report.ipynb'
    input_notebook_sha=sha(path)
    nb=nbformat.read(path,as_version=4)
    print('Executing full notebook from raw in a fresh kernel...',flush=True)
    cell_clock=[time.perf_counter()]
    def completed_cell(**kw):
        now=time.perf_counter()
        print(f"Executed NB[{kw['cell_index']}] in {now-cell_clock[0]:.1f}s",flush=True)
        cell_clock[0]=now
    NotebookClient(nb,timeout=1200,kernel_name=args.kernel,resources={'metadata':{'path':str(ROOT)}},
                   on_cell_executed=completed_cell).execute()
    if sha(path)!=input_notebook_sha:
        raise RuntimeError('Notebook changed on disk during execution; refusing to overwrite user changes')
    nb.metadata['research_revision'].update(stage='P6',status='complete; working-model limitations disclosed',
                                          remaining_stored_outputs='fresh full raw execution')
    path.write_text(nbformat.writes(nb),encoding='utf-8',newline='\n')
    print('Running complete test suite with full-cohort validation...',flush=True)
    env=os.environ.copy();env['COX_FULL_COHORT_TESTS']='1'
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,
                         env=env,capture_output=True,text=True,encoding='utf-8')
    print(tests.stdout+tests.stderr,flush=True)
    if tests.returncode:raise RuntimeError('Unit tests failed; verification not updated')
    number=int(re.search(r'Ran (\d+) tests',tests.stdout+tests.stderr).group(1))
    test_record={'run':number,'failures':0,'errors':0,'skipped':0,'command':'python -m unittest discover -s tests -v',
                 'full_raw_enabled':True,'output':tests.stdout+tests.stderr}
    if 'skipped=' in tests.stdout+tests.stderr:raise RuntimeError('Unexpected skipped tests')
    html,_=HTMLExporter().from_notebook_node(nb)
    html_path=ROOT/'artifacts/Cox_PH_Internal_Report.html'
    html_path.write_text(html,encoding='utf-8',newline='\n')
    manifest=json.loads((ROOT/'artifacts/run_manifest.json').read_text(encoding='utf-8'))
    figures=json.loads((ROOT/'artifacts/figures_manifest.json').read_text(encoding='utf-8'))
    assert {p.relative_to(ROOT).as_posix() for p in (ROOT/'outputs').rglob('*.png')}=={f['path'] for f in figures}
    assert {p.stem for p in (ROOT/'artifacts/tables').glob('*.csv')}==set(manifest['required_tables'])
    hashes={sha(ROOT/f['path']):f['path'] for f in figures}
    embedded=[];counts=[]
    for i,cell in enumerate(nb.cells):
        if cell.cell_type=='code':
            counts.append(cell.execution_count)
            assert all(o.output_type!='error' for o in cell.get('outputs',[])),i
        for output in cell.get('outputs',[]):
            png=output.get('data',{}).get('image/png')
            if png:
                digest=hashlib.sha256(base64.b64decode(png)).hexdigest()
                assert digest in hashes,(i,'embedded PNG mismatch')
                embedded.append({'cell':i,'path':hashes[digest],'sha256':digest})
    assert counts==list(range(1,len(counts)+1))
    assert {x['path'] for x in embedded}=={f['path'] for f in figures}
    assert all(sha(ROOT/'src'/name)==digest for name,digest in manifest['source_code_sha256'].items())
    assert all(sha(ROOT/'data'/name)==info['sha256'] for name,info in manifest['raw_files'].items())
    individual=pd.read_csv(ROOT/'artifacts/tables/individual_predictions.csv')
    assert len(individual)==188976 and individual.loan_identifier.nunique()==47244
    assert ((individual[['predicted_CIF_1','predicted_CIF_2','survival']].sum(axis=1)-1).abs()<1e-10).all()
    assert not individual.duplicated(['loan_identifier','horizon']).any()
    # Checkpoint provenance is completed before verification hashes are recorded.
    cp=json.loads(cp_path.read_text(encoding='utf-8'))
    with (ROOT/'artifacts/research_bundle.pkl').open('rb') as stream:bundle=pickle.load(stream)
    # Refresh earlier checkpoint result views from this actual full raw run.
    dev,tv=bundle['development'],bundle['time_development']
    cp['stages']['P3'].update(selected=dev['selected'],comparison=dev['comparison'].to_dict('records'),
        LR=dev['likelihood_ratios'].to_dict('records'),paired_comparisons=dev['paired_comparisons'].to_dict('records'),
        reproduced_in_P6_UTC=manifest['run_utc'])
    cp['stages']['P4'].update(selected=tv['selected'],comparisons=tv['comparisons'].to_dict('records'),
        evidence=tv['evidence'].to_dict('records'),reproduced_in_P6_UTC=manifest['run_utc'])
    p5=cp['stages']['P5']
    p5.setdefault('original_selection_lock_UTC',p5['specification_locked_before_holdout_UTC'])
    p5.update(metrics=bundle['validation']['metrics'].to_dict('records'),
        concordance=bundle['concordance'],baseline_final_paired_Brier=bundle['baseline_final_paired_Brier'],
        direct_PI_sensitivity=bundle['cif_integration_sensitivity'],
        common_admin_support_120=bundle['administrative_support_sensitivity']['metrics'].to_dict('records'),
        specification_locked_before_holdout_UTC=manifest['selection_locked_before_holdout_UTC'],
        reproduced_in_P6_UTC=manifest['run_utc'])
    cp.update(current_stage='P6',status='COMPLETE',P3_authorized=True)
    cp['stages']['P6']={'status':'PASS WITH LIMITATIONS','completed_utc':datetime.now(timezone.utc).isoformat(),
        'fresh_code_cells':len(counts),'tests_run':number,'figures':len(figures),
        'required_tables':manifest['required_tables'],'baseline_archive_byte_identical':True,
        'single_command':'python src/reproduce.py','verification':'artifacts/verification.json',
        'revision_inventory':revision_inventory(nb,cp['baseline_commit']),
        'remaining_limits':'non-PH, calibration slope, observed-only scope, conditional uncertainty, reused holdout, no external generalization'}
    write_json(cp_path,cp)
    render_checkpoints(cp,manifest,test_record)
    artifact_paths=[path,html_path,cp_path,ROOT/'README.md',ROOT/'analysis_contract.md',ROOT/'stage_report.md']
    artifact_paths.extend(p for p in (ROOT/'artifacts').rglob('*') if p.is_file() and p.name!='verification.json')
    artifact_paths.extend(ROOT/f['path'] for f in figures)
    artifact_paths.extend((ROOT/'src').glob('*.py'))
    artifact_paths.extend((ROOT/'tests').glob('*.py'))
    artifact_paths.extend(ROOT/name for name in ['requirements.txt','.gitattributes','.gitignore'])
    verification={'verified_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS',
        'notebook_sha256':sha(path),'HTML_sha256':sha(html_path),'executed_code_cells':len(counts),
        'execution_counts':counts,'embedded_figures':embedded,'unit_tests':test_record,
        'baseline_archive_sha256':sha(frozen),'model_specification':manifest['specification'],
        'source_code_sha256':manifest['source_code_sha256'],'raw_files':manifest['raw_files'],
        'revision_inventory':cp['stages']['P6']['revision_inventory'],
        'individual_rows':len(individual),'individual_loans':int(individual.loan_identifier.nunique()),
        'artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(artifact_paths))},
        'checks':{'full_execution':True,'no_error_cells':True,'all_figures_embedded_and_current':True,
                  'expected_table_inventory':True,'source_and_raw_hash_match':True,'no_selection_on_legacy_holdout':manifest['checks']['no_selection_on_legacy_test']}}
    write_json(ROOT/'artifacts/verification.json',verification)
    print(f'PASS: {len(counts)} code cells, {number} tests, {len(figures)} figures, {len(manifest["required_tables"])} tables.',flush=True)


if __name__=='__main__':main()
