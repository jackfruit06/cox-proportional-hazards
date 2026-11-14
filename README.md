# Cox PH — Freddie Mac Sample 2015

Nghiên cứu độc lập về thời gian đến **lần đầu quan sát quá hạn từ 90 ngày trở lên**,
có competing exit là RA hoặc Zero Balance hợp lệ xảy ra trước đó (primary ưu tiên
competing exit khi cùng tháng). Đây là phân tích
thống kê, không phải hệ thống chấm điểm, PD theo quy định, hoặc ước lượng tổn thất.

Điểm đọc/chạy chính: [Cox_PH_Internal_Report.ipynb](notebooks/Cox_PH_Internal_Report.ipynb).
Đọc báo cáo revision đã chạy: [bản HTML](artifacts/Cox_PH_Internal_Report.html).
Thiết kế được ghi tại [analysis_contract.md](analysis_contract.md); nhật ký từng
giai đoạn tại [stage_report.md](stage_report.md).

## Final working specification và checkpoint P1–P6

P1 được nghiệm thu; execution-policy update ủy quyền triển khai tuần tự đến P6
và chọn candidate bằng original train/internal validation. Final là
**B_FICO_RCS / static**, hai Cox với **18 tham số mỗi cause**. FICO dùng RCS ba
knots **683/760/803**, học ở train quantiles 10/50/90%. A tuyến tính, C bổ sung
LTV/UPB RCS Cause 2 và hai targeted time effects đã được so sánh; không giữ thêm
complexity khi không đạt OOF/predictive criteria. Xem [checkpoint JSON](artifacts/revision_checkpoint.json)
và [lịch sử stage](stage_report.md) để đọc rule, bằng chứng và giới hạn.

Baseline tuyến tính được cố định tại commit
`4b31e25bb55bca89fdb9aca683c4c26031b36c04`. Code/tables/HTML/figures của baseline
còn trong commit đó; toàn fitted baseline bundle
được giữ byte-identical tại [baseline_bundle.pkl](artifacts/baseline_bundle.pkl).
Các bảng/PNG/HTML/model bundle hiện hành thuộc revision và được tạo lại từ raw.
Notebook tự thực hiện development/selection; không cần đọc fitted model cũ.
[verification.json](artifacts/verification.json) được ghi **sau execute/save/export/tests**,
không dùng verification baseline để chứng minh notebook mới.

**Giới hạn chính:** PH vẫn bị bác bỏ, calibration slopes tại các mốc dài vẫn >1;
đây là working model, không phải true constant HR đã được xác nhận. Legacy test
đã reused; không có external generalization, causal effect, lifetime default hoặc loss claim.

Kết quả và dự báo chính:

- [Hệ số/basis CI](artifacts/tables/cox_coefficients.csv), [joint FICO HR contrasts](artifacts/tables/hazard_contrasts.csv).
- [Primary validation](artifacts/tables/primary_validation.csv), [calibration/CI methods](artifacts/tables/primary_calibration.csv), [landmark 12](artifacts/tables/landmark12_validation.csv).
- [Train-only model comparison](artifacts/tables/model_development_comparison.csv), [sensitivities](artifacts/tables/sensitivity_metrics.csv).
- [Individual CIF/ranking](artifacts/tables/individual_predictions.csv): **188.976 rows / 47.244 loans L=0**, tuổi 12/36/60/120, F1/F2/S, rank và training/reused-holdout role. **2.736 late entrants không thuộc FPD-origin ranking này.**

Không tạo lại thư mục/báo cáo audit. `RESEARCH_REVISION_DECISION_REPORT.md` ngoài
repository vẫn là lịch sử audit; các file bằng chứng audit đã được xóa theo yêu cầu.

### Chạy riêng validation P2, không fit Cox

Từ thư mục repository:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:COX_FULL_COHORT_TESTS='1'
python tests/test_cohort_risksets.py -v
```

Lệnh này chạy 37 tests: 14 synthetic P2, 9 full-cohort P2, 5 P1 regressions và
9 method tests gốc không cần fit. Chặn `PHReg.fit()` cho toàn suite. Hai tests
gốc có fit dữ liệu tổng hợp được giữ nguyên nhưng không nằm trong suite P2;
không gọi đây là chạy toàn bộ tests của mọi giai đoạn. Nếu bỏ biến opt-in raw,
9 full-cohort tests được skip; lượt checkpoint hiện tại đã bật và không skip.
Parser raw dùng lại `audit_sources()`; raw hash và immutable baseline archive
được kiểm tra. Lệnh P2 không ghi lại model, CSV, PNG, HTML hoặc fitted bundle.

## Nguồn và tính độc lập

Ba file raw được sao chép nguyên trạng từ `D:\HUS\TUD\data`; SHA-256 của bản sao
và nguồn đã được đối chiếu khi bàn giao. Notebook kiểm tra lại raw với
`artifacts/source_manifest.json`, nên bản clone không cần thư mục dữ liệu bên
ngoài repository. Không đọc code, cohort, mô hình, biểu đồ
hoặc kết quả từ `v1`, `v2` hay dự án Cox khác. Mọi con số được tính lại từ raw.

- `sample_orig_2015.txt`: 31 trường phân tách `|`, không header.
- `sample_perf_2015.txt`: 35 trường phân tách `|`, không header.
- `Freddie_Mac_Sample_2015_Data_Dictionary_Release47.xlsx`: nguồn ánh xạ vị trí,
  sentinel và quy tắc vintage 2015. Không lấy số liệu validation ghi sẵn trong
  workbook làm kết quả thực nghiệm mới.

## Cấu trúc

```text
COX_PH_v3/
  data/                          # ba bản sao raw, giữ nguyên
  src/
    preprocessing.py             # raw audit, cohort, split, encoder train-only
    cox_model.py                 # hai cause-specific Cox, baseline, CIF
    evaluation.py                # diagnostics, AJ, IPCW, calibration, bootstrap
    reproduce.py                 # execute/save/test/export/final verification
  notebooks/
    Cox_PH_Internal_Report.ipynb  # toàn bộ EDA, hình, lập luận và kết luận
  outputs/
    eda/                         # chỉ PNG
    model/
    diagnostics/
    evaluation/
  artifacts/
    tables/                      # chỉ các bảng cuối phục vụ báo cáo
    research_bundle.pkl          # cohort, split, encoder, models, kết quả
    baseline_bundle.pkl          # original ZB-only baseline, byte-identical
    revision_checkpoint.json     # lịch sử quyết định/checks P1–P6
    data_audit.json               # audit raw và SHA-256
    source_manifest.json          # SHA-256 nguồn đã xác minh, dùng khi clone
    preprocessing.json           # tham số fit trên train và danh sách X
    figures_manifest.json        # mỗi hình gắn với mục đích phân tích
    run_manifest.json            # seed, versions, source hashes, assertion checks
  tests/test_methods.py          # kiểm thử công thức bằng dữ liệu tổng hợp nhỏ
  tests/test_cohort_risksets.py  # P2 no-fit suite, opt-in full raw validation
  tests/test_revision_models.py  # RCS, time effects, CIF/CI/AUC/ranking
  analysis_contract.md
  stage_report.md
  requirements.txt
```

Không cần dữ liệu đã xử lý hoặc pickle từ lần chạy trước để thực thi notebook.
Các module không vẽ hình; phần trình bày được viết trực tiếp trong notebook.

## Môi trường và cách chạy

Clone đầy đủ, bao gồm tệp performance được lưu qua Git LFS:

```powershell
git lfs install
git clone https://github.com/jackfruit06/cox-proportional-hazards.git
cd cox-proportional-hazards
git lfs pull
```

Git LFS cần có trên máy trước khi tải raw. Repository giữ toàn bộ v3, gồm dữ
liệu gốc, notebook đã chạy, hình, bảng và mô hình; chỉ bỏ cache/môi trường ảo.
`.gitattributes` giữ nguyên byte của các tệp để bảo toàn SHA-256.

Đã dùng Python 3.12.7 trên Windows. Các phiên bản thực tế có trong
`artifacts/run_manifest.json`. Từ thư mục dự án, có thể tạo môi trường riêng:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m ipykernel install --user --name cox-ph-v3 --display-name "Cox PH v3"
```

Trong VS Code/Jupyter, mở notebook, chọn kernel đã cài rồi **Restart Kernel and
Run All**. Chạy từ thư mục dự án hoặc thư mục `notebooks/`. Không chạy các cell
rời rạc rồi dùng biến từ phiên trước. Không cần tải dữ liệu qua mạng khi chạy.

Cách chạy đầy đủ gồm notebook, tests, HTML và verification:

```powershell
$env:PYTHONUTF8='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python src/reproduce.py --kernel cox-ph-v3
```

Nếu dùng kernel `python3` tương ứng requirements hiện tại: `python src/reproduce.py`.
Chạy Run All trong IDE chỉ tạo numerical outputs; dùng versioned runner để có
HTML và final verification đồng bộ. Full tests bật raw P2 checks; seed/versions/
sources/specification và input hashes có trong manifest.
Trên Windows, lỗi Jupyter `SetFileSecurity: Access is denied` có thể do sandbox
chặn tạo connection file, không phải lỗi phân tích; chạy kernel ở môi trường
Windows thông thường có quyền ghi runtime directory. Không cần nới lỏng bảo vệ
connection file trong mã dự án.

## Quyết định phương pháp

- Cause 1: delinquency số hợp lệ >=03, lần đầu **quan sát**; RA không là 90+.
  Theo hướng dẫn, 99 là mức trần số tháng quá hạn, XX là missing. File raw này
  không có 99/XX. Không đồng nhất 90+ với legal default hoặc realized loss.
- Cause 2: một nhóm Zero Balance 01/02/03/09/15/16/96; giữ mã gốc, không coi các
  cách kết thúc là tương đương về kinh tế; revision bổ sung RA vào broader
  competing exit, lấy tháng sớm nhất giữa RA/ZB. Giữ `event_source=RA/ZB`,
  `terminal_month`, các ngày/mã gốc; RA/ZB cùng tháng dùng nhãn ZB và cờ
  `zb_ra_same_month`, không suy thứ tự ngày. Primary ưu tiên Cause 2 nếu cùng
  tháng numeric 90+. Numeric 90+ trước RA vẫn là Cause 1.
- FPD là đầu tháng tuổi 0; exit cuối tháng `delta+1`, entry
  `max(0,first_report−FPD)`; risk set `L<t<=T`. Không dùng Loan Age làm clock.
- Primary lấy Zero Balance Effective Date; đối chiếu Period và có sensitivity
  đổi clock. Không tự dịch delinquency theo reporting lag.
- Reused internal holdout 75/25 ở cấp loan, stratify J; seed 20261114. Median,
  reference, nhóm hiếm và missing indicators chỉ học trên train.
- Hai Cox không penalize, Breslow ties, public statsmodels PHReg. Baseline,
  score và information được tính lại/đối chiếu với cùng risk set.
- CIF tích hợp hai hazard bằng exponential allocation từng bước; kiểm tra
  hữu hạn, monotonicity, miền [0,1], `S+F1+F2=1`. Không lấy `1-S_cause1` làm CIF.
- Formal PH time-interaction score test có delayed entry, scaled Schoenfeld,
  martingale và one-step influence; không dùng private API thư viện.
- C-index là cause-specific comparable-pair concordance trên toàn test, có entry;
  competing event censor tại T. Không gọi nó là C-index của CIF hay IPCW C-index.
- Primary CIF validation dùng test **L=0** để đồng nhất FPD origin; landmark 12
  bổ sung người được quan sát và event-free ở tuổi 12, kể cả late entrants đủ điều kiện.
- Calibration dùng AJ incidence/quintile và IPCW logistic intercept/slope;
  CIF Brier dùng train reverse KM, competing outcome là known zero. Benchmark
  là AJ train cùng subgroup và cùng horizon. Không báo IBS từ bốn mốc rời rạc.
- Bootstrap test-loan 200 lần, seed 20261115, percentile CI, giữ nguyên model/G.
  Không tuyên bố CI bao gồm bất định huấn luyện.
- Development: 3-fold train-only A/B/C, knots học lại từng fold; final giữ FICO
  RCS cả hai causes. Time experiments dùng đúng event-time design, analytic
  score/Hessian và time-specific CIF, nhưng không được giữ vì OOF không cải thiện.
- Sensitivity: same-month priority, ZB clock, RA policy cũ, linear baseline trên
  cùng revised outcomes, direct product-integral và common administrative support
  tại 120. Không chọn specification hoặc estimator bằng reused-test metrics.
- CIF C/D AUC với IPCW đo ranking riêng; joint calibration khác offset CITL.
  Fully observed zero-event boundary dùng exact interval, không CI [0,0].

## Đọc kết quả và giới hạn

Các bảng **revision** trong `artifacts/tables/` gồm flow/exclusions, thống kê mô tả, hệ số
và HR/CI, PH/functional-form tests, concordance, primary/landmark validation,
calibration và sensitivity. Tất cả hình PNG có đường dẫn/mục đích trong manifest.

PH bị vi phạm ở cả hai cause ngay cả sau FICO RCS. Đây là kết quả
nghiên cứu cần trình bày, không phải lỗi bị che giấu hay lý do gọi model đã
đạt giả định. HR không đổi và CIF từ primary có giới hạn đặc tả. FICO ghi khi
acquisition không luôn chắc chắn là score đo đúng ngày origination.

Không có performance trước acquisition; giả định entry độc lập chưa được chứng
minh. Primary CIF validation không đại diện mọi late entrant. Gap, reporting
cycles, missing data, pooling nhóm hiếm và marginal censoring model tạo thêm
giới hạn. Internal holdout một vintage không phải external validation. Xem toàn
bộ diễn giải, CI và giới hạn ở cuối notebook trước khi sử dụng kết quả.

## Nguồn phương pháp

[Freddie Mac User Guide](https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf),
[BMJ 2022 — competing-risk validation](https://www.bmj.com/content/377/bmj-2021-069249),
[BMJ 2024 — model development](https://pmc.ncbi.nlm.nih.gov/articles/PMC11369751/),
[statsmodels PHReg 0.14.4](https://www.statsmodels.org/v0.14.4/generated/statsmodels.duration.hazard_regression.PHReg.html),
[Grambsch & Therneau 1994](https://doi.org/10.1093/biomet/81.3.515).
[Harrell — RCS](https://hbiostat.org/course/rms/index.html),
[R survival — direct/exponential CIF](https://www.stat.ethz.ch/R-manual/R-devel/library/survival/html/survfit.coxph.html).
Nguồn web đối chiếu ngày 14/11/2026; Freddie Mac guide hiện là Release 48,
schema thực thi vẫn là workbook Release 47 được cung cấp.
