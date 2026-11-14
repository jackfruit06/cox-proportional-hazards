# Analysis contract — COX_PH_v3

Thiết kế baseline được ghi ngày 14/11/2026, trước fit và đánh giá test của bản gốc.
Revision P1 ngày 14/11/2026 được thực hiện **sau khi test đã được xem** trong
nghiên cứu và audit; không mô tả kế hoạch sửa này là preregistered trước test.
Dự án được viết mới; chỉ đọc ba nguồn gốc trong `data/`. Không nhập code,
bảng phân tích hay kết quả từ `v1`, `v2` hoặc repository Cox khác.

P1 đã triển khai định nghĩa OBSERVED và BROADER COMPETING EXIT được duyệt.
P1 đã được researcher nghiệm thu. Execution-policy update cho phép tự chuyển
P2–P6 và chọn candidate theo train/internal-validation criteria. Checkpoint ở
`stage_report.md` và `artifacts/revision_checkpoint.json`; không đổi event policy
P1. P2 invariant tests không xác nhận PH/truncation độc lập đúng.
Commit baseline bất biến: `4b31e25bb55bca89fdb9aca683c4c26031b36c04`.
Original ZB-only baseline được giữ byte-identical trong `artifacts/baseline_bundle.pkl`
và Git commit đó. Các outputs hiện hành được tái tạo cho final working specification
**B_FICO_RCS / static**, 18 parameters mỗi cause; không có external validation.

## Câu hỏi và kết cục

Nghiên cứu thời gian đến **lần đầu quan sát trong performance history được cung cấp**
Current Loan Delinquency Status
số từ 03 đến 99. Hướng dẫn Freddie Mac hiện hành xác định 99 là mức trần số
tháng quá hạn, **không phải sentinel**; workbook xác định XX là không có dữ liệu,
RA là REO acquisition. Không có 99 hoặc XX trong file đang kiểm tra. RA không
được đổi thành 90+. Không gọi kết cục này là legal default, realized loss hoặc
regulatory PD. FICO được Freddie Mac ghi nhận khi acquisition, thường là điểm
origination; không khẳng định mọi giá trị chắc chắn đo đúng ngày origination.

Cause 2 là **broader competing exit khỏi observable borrower-risk state**:
tháng sớm nhất giữa RA (REO Acquisition) và Zero Balance hợp lệ với mã
01, 02, 03, 09, 15, 16, 96. RA không tương đương ZB hoặc numeric 90+.
So sánh tháng Cause 2 với tháng numeric 90+ đầu tiên: tháng sớm hơn thắng;
nếu cùng tháng, primary vẫn ưu tiên Cause 2. Cause 1 trước RA được giữ nguyên.
Cause 0 là right censoring ở tháng báo cáo cuối khi chưa có event, không phải payoff.

Primary dùng Period cho numeric 90+ và RA, **Zero Balance Effective Date**
cho ZB. `terminal_month` là tháng của outcome thắng; `event_source` nhận
`NUMERIC_90`, `RA`, `ZB` hoặc `CENSOR`. `competing_month` giữ min(RA,ZB).
Nếu RA và ZB cùng tháng, source dùng `ZB` và cờ `zb_ra_same_month=True`;
đây chỉ là quy ước provenance, không khẳng định ZB xảy ra trước RA trong tháng.
Giữ `first_ra_month`, `event_month`, hai tháng ZB và `zb_code` nguyên trạng,
kể cả metadata xảy ra sau terminal event. Mỗi loan chỉ có một outcome.
Các terminal trước observation/FPD vẫn qua exclusions cũ, không tự bỏ candidate
sớm để chọn một event muộn hơn và tạo survivor bias.

Hai sensitivity thời gian vẫn là ưu tiên Cause 1 và dùng Period cho ZB;
clock của RA luôn là Period. Sensitivity refit giữ split ID/encoder/knots cố định;
không chọn model theo legacy test. Thêm ZB-only RA-policy sensitivity có nhãn
riêng, không dùng nó thay approved primary.

## Đồng hồ tháng và delayed entry

First Payment Date (FPD) là đầu tháng tuổi 0. Bản ghi ở tháng cách FPD `a` tháng
đại diện khoảng tháng `(a,a+1]`; event/censoring đặt cuối tháng: `T=a+1`.
`L=max(0,first_report_month-FPD)`; risk set **L < t <= T**.
Các tháng quan sát trước FPD dùng cho audit nhưng không tạo person-time âm.
Không dùng Loan Age làm đồng hồ vì modification có thể reset nó.
Không nhầm tuổi khoản vay `T` với follow-up `T-L`.

Đây là quy ước nhóm thời gian theo tháng, không biết ngày/thứ tự nội tháng.
Không dịch toàn bộ delinquency về tháng trước vì báo cáo có độ trễ. Hướng dẫn
Freddie Mac phân biệt default reporting với investor reporting, đặc biệt trước
05/2019; primary giữ định nghĩa **first observed** trên Period, đối chiếu
Effective Date cho termination. URL guide hiện trả Release 48 (10/2026), trong
khi layout được lấy từ workbook Release 47; không tự đổi layout sang Release 48.

Loại thiếu performance, termination trước quan sát, T<=L, termination trước FPD;
ghi điều kiện tuần tự và số Cause 1/2 bị ảnh hưởng. Không loại riêng vì first UPB=0,
missing covariate hay gap. Gaps được giữ vì estimand là first observed 90+;
không suy diễn không có quá hạn trong tháng không quan sát. Ghi rõ nguy cơ bỏ sót
và sai số thời điểm. Không có lịch sử trước acquisition; độc lập của left truncation
là giả định chưa kiểm chứng, kể cả với người có L=0.

## Preprocessing và split

Random internal holdout 75%/25%, stratify J ở cấp loan, seed=20261114.
P1 giữ nguyên train/test IDs của baseline. Đây là **reused internal holdout**,
đã được xem trước revision. Chỉ train/internal validation được dùng chọn knots,
functional form, time effects hoặc độ phức tạp; test chỉ dùng mô tả cuối.
Một vintage 2015 không được diễn giải là pristine external/temporal validation.
Không đưa biến performance, assistance, modification, current balance/rate vào X.
Median và missing indicators, reference/modal categories, nhóm hiếm <1%, và bỏ
biến hằng được học chỉ từ train. Không loại borrower vì missing covariates.
Chỉ thêm missing indicator khi train có >=50 missing và >=5 biến cố mỗi nguyên
nhân trong nhóm missing để tránh separation; trường hợp ít hơn chỉ median impute.
Scale: FICO/50 điểm; DTI,LTV,MI/10 điểm phần trăm; rate/1 điểm phần trăm;
UPB/100.000 USD; term/120 tháng. Categorical: occupancy, purpose, property,
first homebuyer, units, borrowers. CLTV khảo sát và đối chiếu LTV; primary chọn
LTV để tránh đưa đồng thời hai tỷ lệ gần trùng nhau. VantageScore toàn sentinel
không dùng. Rare/unseen categorical levels ánh xạ về nhóm tham chiếu; báo cáo
đây là giản lược có giới hạn, không phải tham số học từ test.

Không chọn biến bằng univariable p-values hoặc tối ưu test. Tuyến tính log-hazard
được giữ làm **frozen baseline comparator**, không tự coi là final specification.
P3 so sánh A linear, B FICO RCS cả hai causes, C thêm RCS LTV/UPB Cause 2.
Knots học ở quantiles 10/50/90% của đúng training partition, không lấy test hoặc
full cohort; mỗi 3-knot RCS là 2 df tổng, thêm 1 df so với linear. Final FICO
knots trên original train là 683/760/803; fold knots học lại trong mỗi inner train.

3-fold internal CV (seed 20262001), stratify J trong original train, cùng folds
cho mọi candidate; evaluation OOF chỉ trên L=0 ở horizons 36/60/120. Giữ B nếu
FICO LR p<.01 ở cả hai cause và mean OOF Brier không kém A quá một paired fold
SE. Chỉ thêm C nếu relative OOF gain>0.5% và gain>một paired fold SE. Fold SE
là parsimony heuristic với 3 folds, không là formal coverage CI. Kiểm tra thêm
convergence, shape CI, covariance/support; không chọn bằng p-value hoặc test.
B được chọn; C không có predictive gain đủ lớn. Linear tails, median centering
và joint basis contrasts được kiểm tra bằng independent natural-cubic interpolant.

## Ước lượng

Hai cause-specific Cox không penalize, Breslow ties (dữ liệu tháng có ties).
Dùng statsmodels PHReg public API, entry=L+1e-7 để thực thi strict left endpoint
trên đồng hồ tháng nguyên. Nguyên nhân kia được censor ở đúng thời điểm của nó.
Kiểm tra score gần 0, hệ số/covariance hữu hạn và ghi mọi cảnh báo fit.
Tính lại baseline Breslow với đúng risk set; đối chiếu score/Hessian với PHReg.

CIF dùng tích phân theo từng bước hazard: q=1-exp[-(dh1+dh2)], phân bổ q theo
dhk/(dh1+dh2), cập nhật S; không dùng 1-S của Cause 1.
Đây là xấp xỉ hazard hằng trong bước tháng có bảo toàn xác suất; không phải
exact grouped-time likelihood. Có conditioning ở entry hoặc landmark s khi dự báo.
Kiểm tra F1,F2 tăng, trong [0,1], S+F1+F2=1.
Direct product-integral chỉ là sensitivity, với dFk=S_old*hk, S_new=S_old*(1−sum hk)
và guard sum hk<=1. Không clip invalid jumps. Reject horizon trước conditioning time.

## Diagnostics trên train

Tính risk-set moments trực tiếp cho (L,T]. Formal PH score test cho mở rộng
beta(t)=beta+theta*g(t), g(t)=log(t) centered ở các event; dùng Breslow information
và Schur complement để điều chỉnh nuisance beta, global và từng hệ số.
Đây là kiểm định score time interaction dựa trên Schoenfeld, không gọi nhầm
correlation test là formal test và không tuyên bố giống hệt implementation cox.zph.
Scaled Schoenfeld cho hình là nevent*Cov(beta)*(xevent-xbar).
Martingale residual trừ integrated hazard chỉ trên (L,T]. One-step DFBETA từ
score residual để sàng lọc influence; không loại khoản chỉ vì residual lớn.
PH violations và functional form issues là giới hạn của primary Cox, không
tự động chuyển sang một mô hình khác.

P4 thử tối đa một log-time effect/cause: FICO Cause 1, rate Cause 2,
beta_j(t)=beta_j+gamma_j*log(t/36). Fit analytic Breslow likelihood/score/information
với delayed entry, safeguarded Newton; CIF dùng time-specific design tại event t,
không dùng exit T. LR p<.005, HR contrast đổi>=25% giữa tuổi 12/60, reference-profile
CIF delta>=0.002, relative OOF gain>0.5% và gain>paired fold SE đều cần để giữ.
Hai candidates có LR/HR variation nhưng OOF kém và profile delta nhỏ; không giữ TV.
Final static RCS vẫn vi phạm PH, nên HR là working-model association có giới hạn.
Không mô tả việc không chọn TV là xác nhận true PH. Shape/time CI là model-based,
không bao gồm selection uncertainty. Nếu TV được giữ sau này, PH tests chỉ áp
dụng remaining constant terms conditional on TV, không test redundant terms.

## Validation và estimands

Primary C-index: cause-specific hazard Cause 1 trên **toàn test**, event i tại t
so với j có L_j<t<T_j; LP cao hơn ở i là concordant; LP ties được 0.5, outcome
ties không so sánh. Competing event censor tại T. Đây là Harrell-type conditional
comparable-pair concordance, phụ thuộc censoring, **không phải C-index của CIF**,
không tuyên bố đã IPCW cho C-index.
Nếu có TV, score của case/control tính tại cùng event t. CIF discrimination được
báo riêng bằng weighted C/D AUC: cases=Cause 1 by h, controls=not Cause 1 by h,
bao gồm competing events, censoring weights như Brier và score ties=0.5.

Để không trộn dự báo từ FPD với incidence có entry khác nhau, **primary CIF
validation giới hạn test L=0**. Fit vẫn dùng toàn train có delayed entry. Incidence
quan sát là AJ trên cùng subgroup L=0; benchmark AJ train L=0, cùng horizon.
Báo cáo số người bị giới hạn và không ngoại suy đánh giá này cho late entrants.
Phân tích phụ landmark s=12: train/test có L<=12<T, dự báo có điều kiện sống
không event đến 12; tính AJ, censoring G và metric cho thời gian sau s, gồm cả
late entrants đủ điều kiện. Horizon là **tuổi 36,60,120**, không phải 36/60/120
tháng thêm sau landmark.

Horizon candidate 12,36,60,120 tháng được khóa trước test. Chỉ báo cáo khi
train tương ứng có >=200 at risk, >=20 Cause 1 tích lũy và G(h)>0.1.
Nếu test có <20 Cause 1 thì đánh dấu uncertainty, không loại theo kết quả score.

IPCW CIF Brier: (1/n)*sum w_i(h)*(I(T_i<=h,J_i=1)-p_i(h))².
Hai event đều làm status ở h được biết: event trước/tại h dùng 1/G(T_i-),
người T_i>h dùng 1/G(h), Cause 0 trước/tại h trọng số 0.
G là reverse KM **fit trên train của cùng subgroup/landmark**, mọi event J=1/2
được censor trong mô hình censoring; khi ties event/censor, event đứng trước
censor và censor denominator bỏ event cùng tháng. Không coi competing là mất
theo dõi trong Brier. Censor independence/exchangeability và positivity là giả
định, không được chứng minh bằng mô tả. Không gọi complete-case Brier là IPCW.

Calibration: mean prediction, test AJ observed, difference và O/E; quintile
predicted CIF với AJ, IPCW weighted logistic calibration intercept/slope trên
logit(p), dùng cùng weights (competing status=0). Slope/intercept chỉ là calibration
logistic weighted trên horizon, không là hệ số Cox/Fine-Gray. Plot smooth từ
weighted logistic như một tóm tắt tham số; không tuyên bố là nonparametric curve.
Các cột `joint_intercept`/`joint_slope` gọi tên simultaneous fit; `intercept`/`slope`
là aliases tương thích. `CITL_offset` fit slope cố định 1, được bootstrap riêng.
Zero/all-event group có fully observed horizon outcomes dùng exact binomial
boundary interval; nếu còn censoring, flag interval uninformative/NaN, không
trình bày percentile [0,0] như 95% coverage có nghĩa.

Bootstrap 200 lần ở cấp test loan, seed=20261115, percentile 2.5/97.5% CI,
giữ nguyên model, preprocessing và train G; resample test và tính lại test AJ,
mean prediction, Brier, benchmark, calibration và C-index. Báo cáo số thành công.
CI phản ánh test sampling, không bao trùm bất định phát triển mô hình.
Không gọi trapezoid của 4 horizon là Integrated Brier Score; không báo IBS.

Common administrative support sensitivity tại 120 dùng potential cutoff age
**>120** vì censor tại horizon có status unknown theo convention này; đó là
population khác. Không chọn G/CIF method theo test Brier. Individual predictions
cho toàn cohort L=0 ở 12/36/60/120 gồm F1/F2/S và rank, với training/reused-holdout
labels; không xếp lẫn entry-conditional predictions của late entrants từ các origins.

## Reproducibility

`python src/reproduce.py` thực thi notebook từ raw trong kernel mới, save rồi
export HTML, chạy full tests và ghi hashes sau cùng. Không cần fitted bundle cũ
để tính lại. Original baseline archive chỉ phục vụ bảo toàn/comparison, không
là input để fit candidate. Checkpoint P1–P6 dùng cơ chế hiện có, không audit folder.

## Nguồn đối chiếu

- Workbook Release 47 trong data (schema, sentinel, vintage).
- [Freddie Mac General User Guide](https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf), truy cập 14/11/2026, Release 48; phần interpreting data.
- [van Geloven et al., BMJ 2022](https://www.bmj.com/content/377/bmj-2021-069249): AJ, censoring-aware calibration và CIF Brier; không mượn ví dụ/kết quả của bài.
- [Efthimiou et al., BMJ 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11369751/): pre-specification, holdout, preprocessing, uncertainty.
- [statsmodels PHReg 0.14.4](https://www.statsmodels.org/v0.14.4/generated/statsmodels.duration.hazard_regression.PHReg.html): entry và Breslow.
- [Grambsch & Therneau 1994](https://doi.org/10.1093/biomet/81.3.515): weighted residual/time-interaction diagnostics.
- [SAS PHREG score and weighted residual definitions](https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/statug_phreg_a0000000693.htm).
- [Harrell, Regression Modeling Strategies](https://hbiostat.org/course/rms/index.html): restricted cubic splines và linear tails.
- [R survival, survfit.coxph](https://www.stat.ethz.ch/R-manual/R-devel/library/survival/html/survfit.coxph.html): direct và exponential/matrix-exponential constructions.
