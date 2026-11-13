# Analysis contract — COX_PH_v3

Ngày khóa thiết kế: 14/11/2026, trước khi fit mô hình hoặc xem hiệu năng test.
Dự án được viết mới; chỉ đọc ba nguồn gốc trong `data/`. Không nhập code,
bảng phân tích hay kết quả từ `v1`, `v2` hoặc repository Cox khác.

## Câu hỏi và kết cục

Nghiên cứu thời gian đến **lần đầu quan sát** Current Loan Delinquency Status
số từ 03 đến 99. Hướng dẫn Freddie Mac hiện hành xác định 99 là mức trần số
tháng quá hạn, **không phải sentinel**; workbook xác định XX là không có dữ liệu,
RA là REO acquisition. Không có 99 hoặc XX trong file đang kiểm tra. RA không
được đổi thành 90+. Không gọi kết cục này là legal default, realized loss hoặc
regulatory PD. FICO được Freddie Mac ghi nhận khi acquisition, thường là điểm
origination; không khẳng định mọi giá trị chắc chắn đo đúng ngày origination.

Cause 2 gộp mọi mã Zero Balance hợp lệ 01, 02, 03, 09, 15, 16, 96 trước Cause 1.
Giữ mã gốc; đây là các lý do kết thúc khác nhau về kinh tế. Cause 0 là right
censoring ở tháng báo cáo cuối, không đồng nghĩa tất toán.

Primary dùng tháng báo cáo cho 90+, **Zero Balance Effective Date** cho Cause 2.
Nếu cùng tháng, ưu tiên Cause 2. Hai sensitivity bắt buộc: ưu tiên Cause 1;
và dùng tháng báo cáo Zero Balance. Dùng cùng split loan ID, encoder và biến,
refit hai Cox; không chọn mô hình theo hiệu năng test.

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
Một vintage 2015 không được diễn giải là external/temporal validation.
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
là đặc tả cơ sở có kiểm tra residual; khảo sát quadratic score train như một
kiểm tra functional form có mục tiêu. Không thêm splines tự động. Khi chưa có
bằng chứng đủ về đặc tả thay thế, giữ primary và công khai misspecification.

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

Ghi nhận sau diagnostics train: FICO có bằng chứng phi tuyến ở cả hai cause;
thực hiện đúng một sensitivity thêm `(FICO_centered/50)^2` (1 df thêm, tổng
2 df cho FICO), giữ median/scale/covariates/split như primary. Không spline,
không chọn mở rộng bằng test. So sánh nested partial likelihood trên train,
đường HR kết hợp tuyến tính+bậc hai và metric test như phân tích phụ. HR của
riêng hệ số bình phương không phải HR cố định cho mỗi tăng 50 điểm FICO.

## Validation và estimands

Primary C-index: cause-specific hazard Cause 1 trên **toàn test**, event i tại t
so với j có L_j<t<T_j; LP cao hơn ở i là concordant; LP ties được 0.5, outcome
ties không so sánh. Competing event censor tại T. Đây là Harrell-type conditional
comparable-pair concordance, phụ thuộc censoring, **không phải C-index của CIF**,
không tuyên bố đã IPCW cho C-index.

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

Bootstrap 200 lần ở cấp test loan, seed=20261115, percentile 2.5/97.5% CI,
giữ nguyên model, preprocessing và train G; resample test và tính lại test AJ,
mean prediction, Brier, benchmark, calibration và C-index. Báo cáo số thành công.
CI phản ánh test sampling, không bao trùm bất định phát triển mô hình.
Không gọi trapezoid của 4 horizon là Integrated Brier Score; không báo IBS.

## Nguồn đối chiếu

- Workbook Release 47 trong data (schema, sentinel, vintage).
- [Freddie Mac General User Guide](https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf), truy cập 14/11/2026, Release 48; phần interpreting data.
- [van Geloven et al., BMJ 2022](https://www.bmj.com/content/377/bmj-2021-069249): AJ, censoring-aware calibration và CIF Brier; không mượn ví dụ/kết quả của bài.
- [Efthimiou et al., BMJ 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11369751/): pre-specification, holdout, preprocessing, uncertainty.
- [statsmodels PHReg 0.14.4](https://www.statsmodels.org/v0.14.4/generated/statsmodels.duration.hazard_regression.PHReg.html): entry và Breslow.
- [Grambsch & Therneau 1994](https://doi.org/10.1093/biomet/81.3.515): weighted residual/time-interaction diagnostics.
- [SAS PHREG score and weighted residual definitions](https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/statug_phreg_a0000000693.htm).
