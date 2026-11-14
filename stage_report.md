# Báo cáo triển khai theo giai đoạn

**Trạng thái hiện tại: P1–P6 hoàn tất theo execution-policy update; final working model có giới hạn được công khai.**
Các “Giai đoạn 1–7” bên dưới là lịch sử tạo baseline ZB-only, không phải kết quả
của revision. Lịch sử checkpoint revision P1/P2 nằm cuối file; audit lịch sử là
`../RESEARCH_REVISION_DECISION_REPORT.md`, không tạo báo cáo/thư mục audit mới.

Dự án `COX_PH_v3`, 14/11/2026. Chỉ ba nguồn raw được sử dụng lại; không đọc
hoặc sao chép code/kết quả từ các dự án Cox trước.

<!-- FINAL_BASELINE_COMPARISON -->
## So sánh cuối — original ZB-only baseline và revised model

P2 đã được researcher nghiệm thu rõ ràng trong yêu cầu tiếp theo. P3–P6 đã hoàn tất theo chính sách tự triển khai trước lần nghiệm thu này; cập nhật dưới đây chỉ ghi nhận approval và đối chiếu các kết quả đã lưu, không chọn lại/refit model hoặc chạy lại thí nghiệm.

| Quantity | Original ZB-only linear | Linear on accepted P1 cohort | Revised final |
| --- | --- | --- | --- |
| Parameters per cause | 17 / 17 | 17 / 17 | 18 / 18 |
| FICO | Linear | Linear | RCS, knots 683/760/803 |
| Cause 2 source ZB / RA | 37305 / 0 | 37304 / 1 | 37304 / 1 |
| Observed person-months | 3338772 | 3338768 | 3338768 |
| Cause-specific C-index | 0.766500808 | 0.766501200 | 0.765983459 |

| horizon | original_mean_CIF1 | revised_mean_CIF1 | original_Brier | linear_P1_Brier | revised_Brier | original_CIF_AUC | revised_CIF_AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | 0.001397699 | 0.001400892 | 0.001184749 | 0.001184749 | 0.001184637 | 0.7599709 | 0.7665251 |
| 36 | 0.008657014 | 0.00868352 | 0.00912288 | 0.00912288 | 0.009122526 | 0.8024413 | 0.8006253 |
| 60 | 0.02602162 | 0.02616879 | 0.02599356 | 0.02599356 | 0.02596211 | 0.7722077 | 0.7719257 |
| 120 | 0.03989653 | 0.0401818 | 0.03759215 | 0.03759215 | 0.03756168 | 0.745929 | 0.7452719 |

Cột trung gian giữ tuyến tính nhưng dùng cohort P1 để tách thay đổi RA policy khỏi functional form. Original và P1-linear có cùng train/test IDs và hệ số train; Brier ở bốn horizons khớp trong sai số 1e-14 cho dữ liệu này. RA exit duy nhất chuyển từ T=23 về T=19 trên test; điều đó không có nghĩa hai estimands luôn tương đương ở dataset khác.

Giữ nguyên 32 ZB Effective Dates sau reporting month cuối đúng 1 tháng, đúng event clock đã duyệt. Không ép terminal về reporting month. Raw, accepted build_cohort, fitted models, notebook, CSV, PNG và HTML không đổi trong cập nhật nghiệm thu này.

AUC baseline ở bảng này được tính từ frozen predictions và IPCW của baseline gốc, chỉ là point estimate; không dùng chọn model. CIF AUC và cause-specific C-index là hai quantities khác nhau. Final không tốt hơn baseline trên mọi chỉ số. Paired Brier CI của P5 chứa 0 ở cả bốn horizons; PH vẫn bị bác bỏ và calibration slopes dài hạn vẫn >1. Không khẳng định predictive superiority, true PH, calibration đạt hoặc external generalization.

Evidence: `artifacts/baseline_bundle.pkl`, `artifacts/research_bundle.pkl`, `artifacts/revision_checkpoint.json` (`final_baseline_comparison`).
Metadata/comparison refresh UTC: 2026-11-14T02:53:25.386979+00:00. Full numerical run UTC remains 2026-11-14T02:44:56.478823+00:00.
<!-- /FINAL_BASELINE_COMPARISON -->

## Giai đoạn 1 — Khảo sát và thiết kế

- **Đã làm:** kiểm tra Python/thư viện, đọc workbook Release 47, quét toàn bộ
  raw; đối chiếu Freddie Mac guide và nguồn phương pháp. Khóa kết cục, clock,
  delayed entry, exclusions, split, horizons và estimands trước fit.
- **File:** `analysis_contract.md`, ba bản sao `data/`, `artifacts/data_audit.json`,
  phần 1–7 notebook.
- **Đạt:** 50.000 origination × 31 trường, 3.467.166 performance × 35 trường;
  ghép đầy đủ 50.000 IDs, không duplicate ID/tháng, không out-of-order hoặc
  records sau Zero Balance. Không có domain anomaly chưa xử lý ở covariates.
- **Tồn tại:** 36 Period sớm hơn Zero Balance Effective Date một tháng; 2 loan
  có gaps tổng 3 tháng; 24 loan first UPB=0/Zero Balance ở record đầu. 464 loan
  có ít nhất một Loan Age khác calendar FPD age ở record không modification.
  Không dùng Loan Age làm clock. Web guide là Release 48; schema local giữ
  workbook Release 47. Không tuyên bố đã audit nghiệp vụ mọi trường loss/chi phí.
- **Bước tiếp:** xây cohort; không có lỗi cấu trúc nghiêm trọng cần sửa raw.

## Giai đoạn 2 — Xử lý dữ liệu

- **Đã làm:** viết mới `src/preprocessing.py`, tách sentinel/missing, audit
  histories, xây một dòng/một outcome mỗi loan và strict risk set; split 75/25
  ở cấp loan và encoder train-only.
- **File:** module trên; `artifacts/preprocessing.json`; bảng `cohort_flow.csv`,
  `cohort_exclusions.csv`; phần 8, 10–11 notebook.
- **Đạt:** cohort 49.980, loại 20 vì T<=L, đều Cause 2. Giữ Cause 1/2/0 =
  2.019/37.305/10.656; 2.736 late entrants, 30 same-month ties. Train/test
  37.485/12.495, không giao ID; 17 hệ số/cause, preprocessing chỉ fit train.
- **Tồn tại:** không biết trạng thái trong tháng gap. VantageScore toàn sentinel
  không dùng; CLTV/LTV tương quan khoảng 0,949, primary chỉ dùng LTV. Missing
  LTV một loan không đủ cho indicator riêng. Độc lập truncation chưa chứng minh.
- **Bước tiếp:** EDA; không có lỗi thời gian chặn pipeline.

## Giai đoạn 3 — EDA và trực quan hóa

- **Đã làm:** thống kê, missingness, entry/exit/follow-up, biến liên tục/phân loại,
  correlations, đặc trưng theo outcome, AJ hai cause/FICO và outcome counts.
  Inventory hình được ghi trước khi vẽ; hình và giải thích đặt trong notebook.
- **File:** phần 9 notebook; 10 PNG `outputs/eda/`; bảng
  `descriptive_statistics.csv`; `artifacts/figures_manifest.json`.
- **Đạt:** sentinel không lọt vào histogram; AJ dùng delayed entry và competing
  risks, không thay bằng 1−KM. Mọi hình có mục đích, nhãn và đơn vị.
- **Tồn tại:** so sánh theo outcome chỉ mô tả, không có diễn giải nhân quả.
  Full-cohort AJ cần independent truncation nếu diễn giải từ FPD origin.
- **Bước tiếp:** fit hai cause-specific Cox với tập biến đã định.

## Giai đoạn 4 — Mô hình

- **Đã làm:** viết mới `src/cox_model.py`; Cox Breslow public PHReg với entry
  L+1e-7, hệ số/SE/HR/CI/p, baseline và CIF từ cả hai hazards.
- **File:** phần 12–14 notebook; `cox_coefficients.csv`; hình `forest_hr.png`,
  `baseline_hazard.png`, `profile_cif.png` trong `outputs/model/`.
- **Đạt:** hai fit hội tụ không cảnh báo; manual score khớp PHReg và gần 0;
  beta/covariance/predictions hữu hạn. CIF thuộc [0,1], tăng, S+F1+F2=1, kể cả
  conditional theo entry từng khoản.
- **Tồn tại:** Breslow trên tháng và exponential allocation là xấp xỉ đã nêu;
  model-based HR/CI cần kiểm tra giả định.
- **Bước tiếp:** diagnostics train trước khi diễn giải khả năng dự báo.

## Giai đoạn 5 — Diagnostics

- **Đã làm:** `src/evaluation.py` có formal efficient PH score test log-time,
  global/per-variable, Holm; scaled Schoenfeld, entry-correct martingale,
  one-step DFBETA và targeted quadratic scores cho 5 biến liên tục.
- **File:** phần 15 notebook; 3 PNG `outputs/diagnostics/`; bảng
  `ph_score_tests.csv`, `functional_form_scores.csv`.
- **Đạt:** manual information khớp covariance fit; score residual sum gần 0;
  independent finite-difference test khớp PH statistic của likelihood mở rộng.
- **Tồn tại:** cả hai cause vi phạm PH; FICO phi tuyến. Giữ primary theo thiết kế
  và công khai giới hạn; không gọi hội tụ là xác nhận PH. Không loại loan chỉ vì
  influence lớn; one-step DFBETA chưa là refit leave-one-out.
- **Bước tiếp:** đánh giá primary cố định và một sensitivity FICO bậc hai do
  bằng chứng train. Vi phạm giả định là kết quả nghiên cứu, không phải lỗi mã.

## Giai đoạn 6 — Evaluation và sensitivity

- **Đã làm:** cause-specific C-index entry-aware toàn test; CIF validation L=0,
  AJ calibration/quintiles và IPCW logistic slope/intercept; CIF Brier với train
  G, benchmark train AJ cùng subgroup; landmark 12. Bootstrap 200 test-loan,
  seed 20261115, fixed model/encoder/G, percentile CI. Sensitivity đảo ưu tiên
  same-month, Zero Balance clock và một FICO quadratic, cùng pipeline chính.
- **File:** phần 16–20 notebook; 7 PNG `outputs/evaluation/`, hình phụ
  `outputs/model/fico_functional_form.png`; bảng `primary_*`, `landmark12_*`,
  `cause_specific_concordance.csv`, `sensitivity_*` trong `artifacts/tables/`.
- **Đạt:** horizons 12/36/60/120 qua rule hỗ trợ train; competing là known-zero
  trong Brier, censor sớm không bị giả nhãn 0. Benchmark/model cùng test,
  horizon/weights. Mỗi bootstrap có số lần thành công được báo.
- **Tồn tại:** primary CIF validation chỉ cho L=0; landmark khác estimand. Mốc
  12 chỉ có 14 Cause 1 test nên calibration bất định lớn. Marginal G cần censor
  independence/exchangeability. Bootstrap không gồm train uncertainty;
  sensitivity metric chỉ có point estimates, không CI bootstrap riêng.
- **Bước tiếp:** clean notebook execution, kiểm tra hình/bảng/Unicode và README.

## Giai đoạn 7 — Hoàn thiện và xác minh

- **Đã làm:** hoàn thiện `README.md`, `requirements.txt`, báo cáo đầy đủ và kết
  luận từ kết quả chạy; thực thi notebook từ raw trong một kernel mới.
- **File:** notebook đã lưu output, 24 PNG, các bảng cuối;
  `artifacts/research_bundle.pkl`, `run_manifest.json`, `verification.json` và
  bản HTML đọc thuận tiện trong `artifacts/`.
- **Đạt:** xem bằng chứng kiểm thử, không có error cell, số hình và kiểm tra
  Unicode/tái lập tại `artifacts/verification.json`. Raw hashes/source code
  hashes, seed và versions ở `run_manifest.json`. Notebook không cần pickle
  cũ để chạy lại. Các dữ liệu/code phát triển tạm không nằm trong deliverable.
- **Tồn tại:** các giới hạn thống kê ở trên được giữ nguyên trong kết luận;
  không tuyên bố model dùng được cho regulatory PD/credit decisions. Windows
  sandbox từng chặn SetFileSecurity của Jupyter connection file; clean run
  dùng kernel ở môi trường thông thường, không sửa bảo vệ file trong mã dự án.
- **Bước tiếp:** bàn giao. Theo yêu cầu bổ sung, xuất bản toàn bộ v3 lên
  `jackfruit06/cox-proportional-hazards`, bao gồm raw và mô hình. File performance
  dùng Git LFS; manifest SHA-256 giúp notebook chạy độc lập sau clone. Không có
  API, dashboard hoặc slide ngoài phạm vi.

---

## Stage: P1 — Research contract and broader RA competing exit

### 1. Work completed

- Kiểm tra trước sửa: branch `main`, commit
  `4b31e25bb55bca89fdb9aca683c4c26031b36c04`, working tree sạch. Không có thay
  đổi người dùng cần tránh ghi đè. Audit report khớp bản lịch sử đã bàn giao.
- Sửa `src/preprocessing.py::build_cohort()`: Cause 2 ở min(first RA, selected
  ZB clock), so với first observed numeric 90+; primary vẫn ưu tiên Cause 2 khi
  cùng tháng. Thêm `terminal_month`, `event_source`, `competing_month`,
  `zb_ra_same_month`; không thay parser raw, clock FPD, L, exclusions hoặc split.
- `event_source` là `NUMERIC_90`, `RA`, `ZB`, `CENSOR`. RA/ZB cùng tháng dùng
  nhãn ZB và cờ co-occurrence, giữ cả ngày/mã gốc; không suy thứ tự ngày. Metadata
  RA/ZB sau endpoint được giữ, không trở thành một outcome thứ hai.
- Sửa `analysis_contract.md`, `README.md`: OBSERVED, BROADER COMPETING EXIT,
  reused internal holdout và hướng IMPROVE; tách approval hướng nghiên cứu khỏi
  approval candidate/final model. Không mô tả revision sau audit là pre-test.
- Sửa notebook **zero-based cells 0, 2, 3, 6, 7, 8, 38**, giữ nguyên 41 cells
  và IDs. Cell 7 kiểm tra source/terminal và hiển thị phân nhóm nguồn Cause 2;
  tạm không xuất flow/exclusions để bảo toàn CSV baseline. Xóa riêng saved output
  cell 7 vì code đã đổi; mọi output cell khác giữ nguyên và có nhãn baseline.
- Thêm 5 test methods P1 vào `tests/test_methods.py`, cập nhật history fixture
  cũ có `first_ra_month`. Không sửa `cox_model.py`, `evaluation.py` hoặc encoder.
- Lưu một checkpoint máy đọc được tại `artifacts/revision_checkpoint.json`;
  dùng file này làm checkpoint revision, không tạo audit directory, script tạm,
  bản sao pipeline hoặc CSV trung gian.

Baseline tuyến tính và kết quả đã lưu được bảo toàn tại commit trên. SHA-256
của notebook baseline là
`e5471eb4457b78df946108d0091094c6350491ce658bb9ecaaf5242d0a6af64b`;
của `src/preprocessing.py` baseline là
`239218eff7285fb19d240f4a2b3df7d58551bb19e4f8fd79dbe787e33ef707df`.
Checkpoint ghi hashes 48 file raw/artifacts/outputs được giữ nguyên. Không refit,
không tính lại metric test, không xuất lại HTML/verification/bundle/model results.
Notebook baseline còn nguyên trong Git tại commit đã ghi, không cần tạo bản sao.

### 2. Tests performed

Lệnh: `python -m unittest discover -s tests -v`.

- **16/16 tests đạt**, không skip/fail: 11 tests gốc và 5 methods P1.
- `test_earliest_exit_and_source`: numeric trước RA; RA trước numeric; RA không
  có numeric/ZB; ZB trước RA/numeric; RA=ZB; không event; đúng source/terminal,
  một loan/một outcome và thời gian tính từ FPD.
- `test_same_month_numeric_priority_and_source_tie`: RA=numeric theo cả hai
  priority, cùng RA/ZB/numeric; không đếm trùng.
- `test_zb_clock_does_not_shift_ra_clock`: đổi effective/report ZB có thể đổi
  event thắng nhưng không dịch Period của RA.
- `test_entry_and_exclusions_use_selected_ra_terminal`: delayed entry và
  follow-up dùng terminal RA mới; terminal trước observation không được bỏ qua
  để giữ một event ZB muộn hơn.
- `test_invalid_policy_is_not_silently_accepted`: từ chối priority/clock sai.
- Chạy lại preprocessing raw bằng parser hiện có: 50.000 origination và
  3.467.166 performance records. Dùng đúng `build_cohort()` từ baseline Git trên
  cùng raw summaries để tái dựng đối chứng; khớp toàn bộ cohort baseline trong
  bundle. Đây là phép kiểm tra P1, không lặp lại toàn bộ methodological audit.
- Before/after có cùng IDs, flow, exclusions, J/L và tie flags; chỉ T/follow-up
  của một loan đổi. Tổng outcome=cohort, unique IDs, T>0, L<T, source đúng J;
  mọi RA loan exit không muộn hơn RA. Train/test IDs khớp chính xác baseline.
- Notebook schema hợp lệ, toàn bộ 23 code cells parse được. Kiểm tra logic cell
  cohort riêng ở mức P1; **không full-execute notebook**, không chạy P2/P3/model.

### 3. Research results

| Quantity | Baseline ZB-only | Revision P1 |
| --- | ---: | ---: |
| Eligible cohort | 49.980 | 49.980 |
| Cause 1 | 2.019 | 2.019 |
| Cause 2 | 37.305 | 37.305 |
| Cause 0 | 10.656 | 10.656 |
| Cause 2 source ZB | 37.305 | 37.304 |
| Cause 2 source RA | 0 | 1 |
| Exclusions | 20 | 20 |
| Late entrants | 2.736 | 2.736 |
| Numeric/competing same-month ties | 30 | 30 |
| Observed follow-up, person-months | 3.338.772 | 3.338.768 |
| Train/test | 37.485 / 12.495 | 37.485 / 12.495 |

**Danh sách thay đổi outcome/time chỉ gồm một loan:**

| Loan ID | FPD / first observed | First numeric 90+ | First RA | ZB report / effective / code | Before | After | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F15Q30019368 (test) | 201509 / 201508 | Không có | 201703 | 201707 / 201707 / 09 | J=2, L=0, T=23, source ZB, terminal 201707 | J=2, L=0, T=19, source RA, terminal 201703 | Dừng risk state tại RA trước ZB; giảm 4 tháng follow-up |

Cả 55 RA loans được xử lý; **54 loans có numeric 90+ trước RA vẫn giữ Cause 1**,
T không đổi. Raw có 567 RA records, không phải 55 events Cause 2. Không có RA/ZB
same-month case trong raw; quy ước provenance tie được kiểm thử tổng hợp. Không
loan nào có hai outcomes hoặc tiếp tục đóng góp risk time sau RA theo endpoint
mới. Thay đổi `event_source` cho các loan khác chỉ là metadata mới, không phải
thay outcome của họ. Count J không đổi không có nghĩa risk process không đổi.

### 4. Issues and limitations

- Lịch sử trước acquisition, gaps, independent truncation và thứ tự ngày vẫn
  chưa quan sát/xác minh được; P1 không giải quyết các giả định này.
- RA là exit khỏi observable borrower-risk state theo quyết định đã duyệt;
  không được mô tả là numeric 90+, default hoặc loss.
- Baseline HR/CIF/metrics vẫn thuộc ZB-only working model và chưa được thay thế.
  Các fit baseline có thể tiếp tục dùng làm comparator; chưa có approved final
  model hoặc kết luận rằng functional form/PH đã được cải thiện.
- Verification/HTML lưu là baseline. Hash notebook đang sửa khác baseline theo
  chủ đích; không tuyên bố verification cũ xác minh revision. P6 mới tái xuất
  đồng bộ sau khi model được duyệt.
- Chưa thực hiện full P2 risk-set validation, P3 nonlinear experiments hoặc các
  bước sau. Không có lỗi kỹ thuật P1 chưa xử lý hay event-policy ambiguity còn
  ẩn trong kết quả hiện tại; nghiệm thu cohort là gate cần researcher xác nhận.

### 5. Acceptance status

**PASS — Safe to proceed.**

Researcher đã nghiệm thu P1 và cho phép thực hiện P2 trong cuộc hội thoại.
Trạng thái ban đầu “RESEARCHER DECISION REQUIRED” được giữ trong lịch sử JSON;
`stages.P1.P2_authorized=true`. Không hỏi lại OBSERVED / BROADER COMPETING EXIT /
IMPROVE đã được duyệt.

### 6. Next-stage recommendation

Gate P1 đã thông qua; P2 được thực hiện theo yêu cầu tiếp theo của researcher.
Kết quả và gate P3 nằm tại checkpoint P2 bên dưới. Không commit hoặc push.

---

## Stage: P2 — Cohort and Risk-Set Validation

### 1. Work completed

- **Nghiệm thu P2 bổ sung:** researcher đã chấp nhận kết quả full-cohort/risk-set và 37 tests P2 trong yêu cầu tiếp theo. P3–P6 đã hoàn tất theo autonomous execution policy; không thay accepted P1/P2 event/risk-set definitions.

- Dùng đúng P1 đã nghiệm thu làm baseline hiện hành. Sáu hashes file P1 khớp
  checkpoint trước khi bắt đầu. Branch vẫn `main`, HEAD vẫn
  `4b31e25bb55bca89fdb9aca683c4c26031b36c04`; các working changes P1 được giữ.
- Xây lại cohort từ toàn raw bằng preprocessing P1. Đối chiếu từng loan với
  scalar candidate ordering độc lập, không gọi `build_cohort()` để tạo expected
  outcome. Không thay event definition, FPD clock, priority hoặc exclusions.
- Kiểm tra từng ô loan×month bằng interval sweep độc lập; đối chiếu risk sets
  Cox, AJ, reverse KM, concordance, residual integration và validation population.
  Kiểm tra entry/exit equality, same-month ties, weighted resampling và RA exits.
- Không gọi Cox optimizer: suite chặn `PHReg.fit()`. Public loglike/score/Hessian
  chỉ được tính tại fixed diagnostic beta. Không chọn model/encoder/knots, fit
  candidate hoặc chấm điểm dự báo legacy holdout.
- Dùng checkpoint hiện có để giữ lịch sử P1/P2; không tạo audit directory/report,
  pipeline thứ hai, bản sao raw hoặc CSV trung gian.

### 2. Files/functions/tests modified

| File | P2 change |
| --- | --- |
| `tests/test_cohort_risksets.py` | Test file chính thức mới: `reference_terminal`, `assert_cohort_policy`, `interval_risk_counts`, `reference_aj`; 14 methods `SyntheticRiskSetsP2`, 9 methods `FullCohortP2`; `NoFitP2Suite`/`p2_suite` cho chạy lại không fit |
| `artifacts/revision_checkpoint.json` | Lưu `stages.P1` và `stages.P2`, nghiệm thu P1, full-cohort/risk-set results, hashes và `P3_authorized=false` |
| `README.md` | Trạng thái P1 accepted/P2 pending acceptance; lệnh chạy lại P2 và scope no-fit tests |
| `analysis_contract.md` | Chỉ cập nhật trạng thái stage, giữ event policy P1 |
| `notebooks/Cox_PH_Internal_Report.ipynb` | Chỉ sửa Markdown cells **0, 8, 38** so với P1 để cập nhật tiến độ/checkpoint; giữ code cells, IDs và outputs P1 |
| `stage_report.md` | Ghi nghiệm thu P1 và checkpoint P2 này |

**Không sửa trong P2:** `src/preprocessing.py`, `src/cox_model.py`,
`src/evaluation.py`, `tests/test_methods.py`, raw và mọi research output baseline.
Accepted P1 preprocessing SHA-256 vẫn là
`9e9bc915a52f4ac84d776ae5c7f55fe2f28cf539537cef8a571a2f4bc3943fa4`.

Synthetic tests bao phủ 27 tổ hợp event ordering; metadata hỏng/ID trùng phải
bị phát hiện; missing-performance/pre-FPD/late-entry exclusions; ZB effective
date sau last Period; strict PHReg adapter; weighted AJ/reverse-KM/concordance
boundaries; RA risk removal; residual integration; IPCW known-zero RA; landmark
eligibility/common-origin filtering trước IPCW. Full tests kiểm tra mọi loan,
mọi tháng, hai causes, train/test IDs và preservation.

**37/37 tests đạt, 0 fail/error/skip**: 23 P2 + 5 P1 + 9 methods gốc không fit.
Hai methods gốc `test_ph_score_matches_finite_difference` và
`test_validation_bootstrap_path` được giữ nguyên nhưng không chạy trong P2 vì
có fit synthetic Cox. Chúng không nằm trong suite 37 tests, không gọi là skipped.

Lệnh chạy lại từ repository:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:COX_FULL_COHORT_TESTS='1'
python tests/test_cohort_risksets.py -v
```

### 3. Full-cohort validation results

| Check | Result |
| --- | ---: |
| Origination / performance rows đọc | 50.000 / 3.467.166 |
| Eligible cohort / excluded | 49.980 / 20 |
| Cause 1 / Cause 2 / censoring | 2.019 / 37.305 / 10.656 |
| Cause 2 source ZB / RA | 37.304 / 1 |
| Late entrants | 2.736 |
| Observed person-months | 3.338.768 |
| Train / test | 37.485 / 12.495 |
| Loan×month memberships checked | 6.747.300 |
| Risk membership violations | 0 |
| Raw/artifacts/outputs baseline hash unchanged | 48 files |

Mỗi retained loan có ID duy nhất, một outcome; J thuộc {0,1,2}, các outcomes
mutually exclusive/exhaustive và cộng đúng cohort. Mọi L/T/terminal/follow-up
hữu hạn; T>0, L>=0, L<T; L=max(0,first−FPD), T=terminal−FPD+1, follow-up=T−L.
`event_source`, `competing_month`, `same_month`, `zb_ra_same_month` và terminal
khớp independent ordering cho từng loan. Không bỏ event sớm để giữ event muộn.

Train/test IDs và thứ tự khớp nguyên vẹn baseline/P1, không giao nhau, hợp lại
đúng cohort. SHA-256 chuỗi IDs theo thứ tự, phân cách newline:

- Train: `836182667928095659d6748509ada43fd3cd6c80d404c37946e2280491dc266b`.
- Test: `5b5635dc858a46406c5f2b98face6be1731a54e7fe0cc6ba2ae853ef0a261864`.

Vẫn chỉ `F15Q30019368` khác baseline ZB-only về T/follow-up: source RA, J=2,
L=0, T=19 thay cho 23. Cả 54 post-numeric-90 RA giữ Cause 1/T cũ. Toàn bộ 55 RA
loans exit không muộn hơn RA; không có risk contribution sau RA.

### 4. Risk-set validation results

| Function / component | Evidence | Assessment |
| --- | --- | --- |
| `cox_model.risk_moments` | Full cohort: 127 Cause 1 event times, 133 Cause 2 times; weighted moments khớp; identity-basis toy xác định từng thành viên | Strict L<t<=T; gồm exit boundary, không gồm entry boundary |
| `cox_model.fit_cause` adapter | Mock dừng ở constructor, kiểm tra entry=L+1e-7, status theo cause/Breslow; public PHReg numerical comparison ở fixed beta | Không gọi fit hoặc đổi fitted model parameters |
| `evaluation.aalen_johansen` | Full monthly AJ khớp interval-sweep AJ, max abs difference **0.0**; weighted/zero-count toy đạt | Risk `(L,T]`, đúng cả hai causes |
| `evaluation.censoring_curve` | Reverse KM full cohort khớp independent counts, max abs difference **0.0**; tied event/censor và entry tests đạt | Event trước censor, censor denominator Y−d1−d2 |
| `evaluation.concordance` | Brute-force synthetic weighted pairs/ties; full-test comparator count **3.650.477** | Control L<t<T loại exit ties có chủ đích; không phải lỗi Cox/AJ risk set |
| `evaluation.diagnostics` | Symmetric fixed-beta toy, covariance từ information, không fit; martingale so với hazard tích lũy chỉ trên `(L,T]` | Không tích lũy hazard trước entry/sau exit |
| `landmark_sample`, `validate`, `ipcw_status` | Boundary eligibility, coordinate shift, mock prediction path; full-cohort subgroup/status checks không chấm điểm model | Primary L=0; landmark L<=12<T rồi L'=0,T'=T−12; RA là known zero ở Cause 1 horizon |

Manual partial likelihood, score/information trên **full revised cohort** so với
public PHReg tại diagnostic columns và beta cố định `[0.1,-0.05]`. Đây chỉ là
numerical probe, không phải fitted/revised research model:

| Cause | Abs likelihood error | Max abs score error | Max abs information error |
| --- | ---: | ---: | ---: |
| 1 | 2.55e-11 | 3.41e-13 | 9.66e-13 |
| 2 | 5.82e-11 | 4.41e-12 | 1.59e-11 |

Grid **1–135** bao gồm toàn support đến T=134 và một tháng sau cùng. Mỗi loan
đóng góp đúng T−L monthly memberships; không membership tại t<=L hoặc t>T.
Risk count lớn nhất 48.296; tại 135 bằng 0. RA loan đặc biệt còn trong risk set
tại event boundary t=19, **không còn tại t=20 trở đi**.

L=0 train/test **35.460/11.784**; landmark 12 train/test **34.063/11.292**, gồm
317 test late entrants đủ điều kiện. `validate()` lọc/chuyển tọa độ trước khi
dùng IPCW; không đưa một hỗn hợp late-entry origins vào absolute-risk validation.
Landmark output giữ date/source metadata gốc nhưng L/T là tọa độ relative cho
metric; không diễn giải đó như một cohort gốc có terminal calendar mới.

Concordance probe gán mọi score bằng nhau để chỉ kiểm tra membership/ties;
probe=0.5 **không phải** model discrimination mới. Không tính lại legacy-holdout
performance, HR, CI hoặc predicted CIF trong P2.

### 5. Issues and limitations

- Không phát hiện defect cần đổi event policy P1, `cox_model.py` hoặc
  `evaluation.py`; không có cohort/risk-set inconsistency chưa giải quyết.
- Lượt đầu có một assertion tôi thêm quá chặt: terminal<=last Period. Có
  **32 ZB exits** có Effective Date sau last reporting month đúng 1 tháng;
  primary đã duyệt dùng Effective Date. Tôi sửa **assertion P2**, không sửa
  policy: chỉ chấp nhận source ZB, terminal đúng effective date được ghi và ZB
  reporting không sau last Period. Đã thêm synthetic regression; lượt cuối đạt.
- Lệnh test hiển thị `ResourceWarning` ở `preprocessing.audit_sources():131`
  do performance reader hiện dùng implicit cleanup thay vì context manager;
  thêm một `DeprecationWarning` từ openpyxl. Đây là housekeeping hiện hữu,
  không làm fail tests hoặc đổi counts/hashes. Không sửa parser P1 trong P2;
  có thể xử lý riêng sau này, không là event-definition defect hoặc gate P3.
- History trước acquisition/gaps, independent truncation/censoring và monthly
  ordering vẫn là giới hạn nghiên cứu; invariant checks không chứng minh các
  giả định không quan sát được này.
- P2 xác minh mechanics, chưa đánh giá PH/functional form/calibration của
  revised final model. Kết quả model hiện lưu vẫn là ZB-only baseline.
- Không full-run notebook revision hoặc export HTML/verification mới. Code
  cells và saved outputs P1 giữ nguyên; notebook chỉ sửa Markdown checkpoint.
- Opt-in raw phải bật để chạy full tests; nếu không, 9 full-cohort tests sẽ
  skip. Lượt P2 hiện tại đã bật, không skip trong 37 tests.

### 6. Acceptance status

**PASS — Safe to proceed.** Cohort/risk-set checks đạt. Execution-policy update
của researcher thay gate xin nghiệm thu bằng tự chuyển stage khi criteria đạt.
Trạng thái chờ review ban đầu được giữ như lịch sử, không coi policy update là
một lần researcher review từng numerical result.

### 7. Recommendation for P3

Đã tự chuyển P3 theo ủy quyền mới; P1 event policy và train/test membership
được giữ. P3–P6 results được ghi bên dưới. Không commit/push.

<!-- REVISION_P3_P6 -->

## Stage: P3 — Nonlinear functional-form experiments

### 1. Work completed

So sánh A linear, B FICO RCS ở cả hai causes, C thêm LTV/UPB RCS Cause 2; 3-fold original-train-only validation. Mỗi fold học encoder/knots riêng. Cùng samples/folds và horizons 36/60/120; legacy test không tham gia selection.

### 2. Tests performed

RCS khớp independent natural-cubic interpolant, linear tails, centering/rank, immutable train-only knots, targeted cause columns và CIF column subsets đều đạt. P6 chạy toàn suite.

### 3. Research results

| specification | cause | parameters | log_likelihood | OOF_Brier_mean_36_60_120 | PH_global_p |
| --- | --- | --- | --- | --- | --- |
| A_linear | 1 | 17 | -14524.14 | 0.0238482 | 2.347896e-10 |
| A_linear | 2 | 17 | -277070.4 | 0.0238482 | 2.73114e-91 |
| B_FICO_RCS | 1 | 18 | -14511.01 | 0.02382381 | 1.029517e-11 |
| B_FICO_RCS | 2 | 18 | -277055 | 0.02382381 | 3.566494e-91 |
| C_targeted_RCS | 1 | 18 | -14511.01 | 0.02382137 | 1.029517e-11 |
| C_targeted_RCS | 2 | 20 | -277040.9 | 0.02382137 | 7.859861e-93 |

| smaller | larger | cause | df | LR | p |
| --- | --- | --- | --- | --- | --- |
| A_linear | B_FICO_RCS | 1 | 1 | 26.27418 | 2.962226e-07 |
| A_linear | B_FICO_RCS | 2 | 1 | 30.77769 | 2.893458e-08 |
| B_FICO_RCS | C_targeted_RCS | 2 | 2 | 28.36691 | 6.921561e-07 |

| smaller | larger | OOF_Brier_gain | paired_fold_SE | relative_gain |
| --- | --- | --- | --- | --- |
| A_linear | B_FICO_RCS | 2.439255e-05 | 3.354617e-06 | 0.001022826 |
| B_FICO_RCS | C_targeted_RCS | 2.431505e-06 | 2.93643e-06 | 0.000102062 |

Selected: **B_FICO_RCS**. Knots 10/50/90% train: FICO 683/760/803; candidate LTV 49/78/95; UPB 90000/200000/396000. A coefficients tái tạo baseline gốc. B cải thiện shape evidence và OOF; C không đạt extra predictive/complexity rule. Shape plots có joint covariance CI trên train 1–99th percentiles.

### 4. Issues and limitations

Remaining PH vẫn bị bác bỏ. Paired fold SE là selection heuristic với 3 folds, không formal CI. Model-based shape CI chưa bao gồm selection uncertainty; không chọn spline chỉ vì p nhỏ.

### 5. Acceptance status

**PASS** — chọn bằng rule đã khóa, không test shopping.

### 6. Next-stage recommendation

Đã chuyển P4 tự động. Files/functions: specification_encoder, internal_folds, functional_experiments; `src/preprocessing.py` (encoder only; accepted build_cohort unchanged), `src/cox_model.py`, `src/evaluation.py`, `tests/test_revision_models.py`, notebook và docs liên quan.

## Stage: P4 — Targeted non-PH assessment

### 1. Work completed

Thử tối đa một time effect/cause: FICO Cause 1, rate Cause 2, beta(t)=beta+gamma log(t/36). Analytic Breslow likelihood/score/information, strict delayed entry, safeguarded Newton fit; predictions dùng actual t. Không dùng exit T làm covariate.

### 2. Tests performed

Gradient/Hessian khớp finite differences; gamma=0 khớp static likelihood; covariance/convergence, time-specific prediction và probability conservation đạt.

### 3. Research results

| specification | OOF_Brier_mean | OOF_gain | paired_fold_SE | relative_gain | max_profile_CIF_difference |
| --- | --- | --- | --- | --- | --- |
| static | 0.02382346 | 0 | 0 | 0 | 0 |
| TV_FICO_C1 | 0.02383438 | -1.091745e-05 | 7.557879e-06 | -0.0004582646 | 0.001330734 |
| TV_rate_C2 | 0.02382692 | -3.457129e-06 | 4.534141e-06 | -0.0001451145 | 0.00084092 |
| TV_both | 0.02383661 | -1.314929e-05 | 6.920199e-06 | -0.0005519471 | 0.001165458 |

| cause | variable | training_LR | p | HR_12 | HR_60 | HR_ratio_change | retain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | classic_fico_score | 48.02845 | 4.200794e-12 | 0.2675636 | 0.4185913 | 1.564455 | False |
| 2 | original_interest_rate | 162.618 | 3.031671e-37 | 2.247681 | 1.546143 | 1.453734 | False |

Selected time structure: **static**. LR/HR changes có evidence, nhưng OOF loss kém và profile CIF deltas dưới 0.002; không giữ thêm complexity. HR(t)+CI figure lưu actual fitted experiments, không mô tả các candidates là primary.

### 4. Issues and limitations

Working RCS vẫn non-PH: Cause 1 global p≈1.03e-11; Cause 2 p≈3.57e-91. Đây là prediction-focused parsimonious working specification, không xác nhận HR thật hằng. Nếu TV được chọn sau này, diagnostics chỉ test remaining constant terms conditional on TV.

### 5. Acceptance status

**PASS WITH LIMITATIONS** — limitations được công khai, không bypass một failed convergence/probability criterion.

### 6. Next-stage recommendation

Đã khóa model rồi chuyển P5. Functions: time_likelihood, fit_time_varying, model_design, time_diagnostics, time_experiments; `src/preprocessing.py` (encoder only; accepted build_cohort unchanged), `src/cox_model.py`, `src/evaluation.py`, `tests/test_revision_models.py`, notebook và docs liên quan.

## Stage: P5 — CIF prediction and final descriptive validation

### 1. Work completed

Giữ exponential-generator primary; direct PI sensitivity có guard combined jump<=1. Reused holdout chỉ dùng sau model lock. Thêm CIF C/D AUC, joint-vs-offset calibration, valid boundary CI và individual CIF/ranks.

### 2. Tests performed

Weighted AUC khớp brute pairs/ties; offset score equation; zero-event fully observed/censoring branches; L=0 ranking; direct/exponential/time-specific mass balance, monotonicity và h>=start guards đạt.

### 3. Research results

| horizon | n | mean_predicted | observed_AJ | brier | benchmark_brier | cif_auc | slope |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | 11784 | 0.001400892 | 0.001188052 | 0.001184637 | 0.001186703 | 0.7665251 | 1.221563 |
| 36 | 11784 | 0.00868352 | 0.009334691 | 0.009122526 | 0.009247906 | 0.8006253 | 1.405641 |
| 60 | 11784 | 0.02616879 | 0.02749491 | 0.02596211 | 0.02674048 | 0.7719257 | 1.253374 |
| 120 | 11784 | 0.0401818 | 0.04056419 | 0.03756168 | 0.03891877 | 0.7452719 | 1.12273 |

| horizon | joint_intercept | joint_slope | CITL_offset | cif_auc_low | cif_auc_high |
| --- | --- | --- | --- | --- | --- |
| 12 | 1.197011 | 1.221563 | -0.1651981 | 0.6172379 | 0.8553611 |
| 36 | 1.812781 | 1.405641 | 0.07352883 | 0.7624413 | 0.8358072 |
| 60 | 0.8738455 | 1.253374 | 0.05188846 | 0.7472957 | 0.793349 |
| 120 | 0.3601636 | 1.12273 | 0.01009297 | 0.7267905 | 0.7656652 |

| horizon | baseline_minus_final_Brier | low | high |
| --- | --- | --- | --- |
| 12 | 1.123349e-07 | -2.908136e-07 | 5.043122e-07 |
| 36 | 3.544309e-07 | -9.988367e-06 | 7.866417e-06 |
| 60 | 3.144205e-05 | -4.734665e-06 | 6.693257e-05 |
| 120 | 3.047351e-05 | -2.441055e-05 | 8.743635e-05 |

Cause-specific C-index=0.765983, fixed-model test CI [0.747822, 0.785699]. CIF AUC là quantity riêng. Individual CSV: 188,976 rows / 47,244 L=0 loans, bốn horizons, F1/F2/S và rank; training/holdout labels rõ.

Quintile zero-event ở 12 tháng có 0/2357, exact upper 95%=0.00156385, thay CI [0,0]. Boundary có censoring không dùng binomial uncritically; flag uninformative/NaN CI. Common administrative support dùng C_admin>120, n=11767 test và 0 early censoring; population khác primary. Priority, ZB clock, RA policy cũ, baseline linear và direct PI đều là declared sensitivities, không dùng chọn model.

### 4. Issues and limitations

Slopes tại 36/60/120 vẫn >1; mean gần AJ không chứng minh calibration toàn risk range. Paired baseline-final Brier CI chứa 0 ở cả bốn horizons, chưa xác lập predictive superiority trên holdout. Test đã reused, CI conditional on fitted model/G; không training/selection uncertainty hoặc external validity. Model-based HR associations không causal/lifetime/default/loss.

### 5. Acceptance status

**PASS WITH LIMITATIONS** — probability/error/ranking mechanics đạt; statistical limitations không bị che.

### 6. Next-stage recommendation

Đã chuyển P6. Functions: predict_cif, validate, cif_auc, calibration_offset, individual_predictions; `src/preprocessing.py` (encoder only; accepted build_cohort unchanged), `src/cox_model.py`, `src/evaluation.py`, `tests/test_revision_models.py`, notebook và docs liên quan.

## Stage: P6 — Reproducibility, export and reporting

### 1. Work completed

Fresh full notebook từ raw; lưu notebook rồi export HTML; tests và artifact verification sau save. Một lệnh versioned `python src/reproduce.py`, không cần script/audit directory tạm. Baseline archive byte-identical; original outputs/code còn trong recorded Git commit.

### 2. Tests performed

**52/52 tests PASS**, full raw P2 enabled, 0 failures/errors/skips. All 23 code cells executed in order; no error outputs. Embedded PNG hashes khớp standalone figures; expected CSV/PNG inventory không có stale extras. Source/raw/notebook/HTML/artifact hashes được ghi sau export tại artifacts/verification.json.

### 3. Research results

Final **B_FICO_RCS / static**, parameters 18/18; FICO knots 683/760/803. Cohort 49980, J1/J2/J0=2019/37305/10656, Cause 2 ZB/RA=37304/1; train/test IDs giữ nguyên. 25 figures, 16 required CSV tables. HR/basis coefficients ở cox_coefficients.csv; joint FICO HR/CI ở hazard_contrasts.csv.

Exact source-function inventory (so với original baseline):

| file | changed definitions |
| --- | --- |
| src/cox_model.py | CoxFit, CoxFit.static_columns, CoxFit.coefficient_table, model_design, reference_rows, hazard_contrasts, fit_models, time_risk_moments, time_likelihood, fit_time_varying, predict_cif |
| src/evaluation.py | specification_encoder, internal_folds, development_losses, functional_experiments, time_diagnostics, time_experiments, concordance, calibration_coefficients, calibration_offset, cif_auc, individual_predictions, validate, diagnostics, functional_form_scores |
| src/preprocessing.py | rcs_component, audit_sources, build_cohort, Encoder, Encoder._matrix, Encoder.with_rcs, Encoder.manifest |
| src/reproduce.py | sha, clean, write_json, markdown_table, revision_inventory, render_checkpoints, main |

Notebook source cells đổi (0-based): [0, 1, 2, 3, 6, 7, 8, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40]. Saved outputs được fresh-execute lại cho mọi code cell.

### 4. Issues and limitations

Full execution/verification không xác nhận independent censoring/truncation, PH, calibration hoặc external generalization. Raw reader ResourceWarning phát hiện ở P2 đã sửa bằng context manager, không đổi accepted build_cohort. Bootstrap-cell timeout đã được xử lý bằng elementwise calibration objective tương đương và regression test; full rerun cuối đạt. Jupyter ACL trong sandbox cần chạy kernel ở môi trường Windows thông thường; không nới lỏng connection security. Upstream openpyxl DeprecationWarning không làm fail tests; không đổi dependencies để che warning.

### 5. Acceptance status

**PASS WITH LIMITATIONS** — P6 technical criteria đạt, research limitations được giữ trong notebook/contract/README.

### 6. Next-stage recommendation

P1–P6 hoàn tất trong scope được ủy quyền. Không commit/push, không sửa raw, không thay endpoint. Muốn external-vintage generalization cần dữ liệu/thiết kế mới; không tự mở rộng task này.
