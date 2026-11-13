# Báo cáo triển khai theo giai đoạn

Dự án `COX_PH_v3`, 14/11/2026. Chỉ ba nguồn raw được sử dụng lại; không đọc
hoặc sao chép code/kết quả từ các dự án Cox trước.

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
