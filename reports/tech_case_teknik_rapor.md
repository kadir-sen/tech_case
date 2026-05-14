# PARA TRANSFERİ SAHTECİLİK TESPİTİ

## 1. Problem Tanımı

Bankanın dijital kanalları (Mobil Şube ve İnternet Bankacılığı) üzerinden gerçekleştirilen para transfer işlemlerinde, müşterinin bilgisi ve isteği dışında gerçekleşen sahte (fraudulent) işlemlerin gerçek zamanlı veya gerçek zamana yakın şekilde tespit edilmesi hedeflenmektedir.

Çıktılar:

- Fraud tespit makine öğrenmesi modeli (eğitim + tahmin kodları)
- Tek işlem skorlayan REST API (FastAPI)
- Uygulanan yöntemleri ve sonuçları içeren bu teknik rapor

---

## 2. Veri Seti Analizi

### 2.1 Genel Bakış

| Özellik | Değer |
|---|---|
| Toplam İşlem | 849,564 |
| Tarih Aralığı | 2022-09-17 → 2024-09-13 (727 gün) |
| Sütun Sayısı | 26 (ham) → 55 (feature engineering sonrası) |
| Fraud İşlem | 5,338 (%0.63) |
| Meşru İşlem | 844,226 (%99.37) |
| Sınıf Dengesizliği | 1:159 |
| Duplicate | BusinessKey'de 0 duplicate |

Veri yükleme:

```python
def load_raw(path=None) -> pd.DataFrame:
    if path is None:
        cfg = load_config("data")
        path = REPO_ROOT / cfg["data"]["path"]
    df = pd.read_parquet(path)
    if "TransactionDate" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["TransactionDate"]):
        df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
    return df
```

```python
df = load_raw()
print(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
print("\nData Types and Null Values")
overview = pd.DataFrame({
    "dtype":   df.dtypes,
    "nulls":   df.isnull().sum(),
    "null_%":  (df.isnull().sum() / len(df) * 100).round(2),
    "nunique": df.nunique()
})
print(overview)
```

Çıktı:

```
Dataset loaded: 849,564 rows x 26 columns

Data Types and Null Values
                          dtype       nulls   null_%   nunique
BusinessKey              object           0    0.00    849,564
AccountNumber             int64           0    0.00    789,861
TransactionDate  datetime64[ns]           0    0.00    849,427
TransactionType          object           0    0.00          3
TransactionChannel       object           0    0.00          1
ReceiverName             object           0    0.00    135,758
SenderName               object           0    0.00      7,242
IsFraudTransaction        int64           0    0.00          2
HasMobileActivationL1H    int64           0    0.00          2
HasMobileActivationL8H    int64           0    0.00          2
DayType                  object     819,083   96.41          2
CustomerName             object           0    0.00      6,894
CustomerSegment          object           0    0.00          9
CustomerAge               int64           0    0.00         72
CustomerTenure          float64           0    0.00        262
CustomerEducation        object      14,132    1.66          9
CustomerProfession       object           0    0.00        112
CustomerMaritalStatus    object         188    0.02          4
CustomerGender           object           0    0.00          2
IsFractionalAmount         bool           0    0.00          2
TransactionAmount       float64           0    0.00    849,564
DeviceModel              object           0    0.00      1,004
DeviceOSName             object           0    0.00          2
DeviceId                 object           0    0.00     13,580
IP_Subnet                object           0    0.00     29,004
UniqueIPCount             int64           0    0.00        681
```

### 2.2 Class Imbalance

Veri setinde **1:159** oranında bir dengesizlik mevcuttur. Tüm işlemlere "fraud değil" denilse %99.37 accuracy elde edilir; bu yüzden değerlendirmede **ROC-AUC, PR-AUC, Precision@k, Recall@k, F1, F2, Brier** metrikleri kullanılmıştır.

```
IsFraudTransaction Distribution
Legit (0):  844,226   (99.37%)
Fraud (1):    5,338    (0.63%)
```

Çeyreklik fraud oranı sabit değildir; zaman içinde drift mevcuttur:

```
2022Q3 |   5,762 |     0  | 0.0000%
2022Q4 |  66,049 |    25  | 0.0379%
2023Q1 |  85,504 |   281  | 0.3286%
2023Q2 |  98,465 |   931  | 0.9455%
2023Q3 | 109,652 | 1,381  | 1.2594%
2023Q4 | 117,063 | 1,272  | 1.0866%
2024Q1 | 122,933 |   884  | 0.7191%
2024Q2 | 129,409 |   415  | 0.3207%
2024Q3 | 114,727 |   149  | 0.1299%
```

### 2.3 Data Analysis

**İşlem Tutarı (TransactionAmount)**

- Sağa çarpık dağılım: medyan ~5,600 TL, maksimum 344,573 TL.
- Fraud işlemlerin ortalaması (~20,545 TL), meşru işlemlerin ortalamasından (~13,430 TL) belirgin biçimde yüksektir.

```
--- Amount stats by fraud label ---
IsFraudTransaction      count           mean          std         min         50%           max
0                  844,226.0000  13,430.8325  19,679.34   500.0000  5,582.5462  344,573.9944
1                    5,338.0000  20,545.4252  29,709.43   544.5979  9,070.2759  321,799.0483
```

**Mobil Aktivasyon (HasMobileActivationL1H / L8H)**

- Son 1 saatte mobil aktivasyon varsa fraud oranı: **%10.99** (aktivasyon yoksa: %0.60)
- Son 8 saatte mobil aktivasyon varsa fraud oranı: **%8.19** (aktivasyon yoksa: %0.59)
- Bu sinyaller hesap ele geçirme (Account Takeover) senaryolarının tipik göstergesidir.

**IP Sinyalleri (UniqueIPCount)**

```
UniqueIPCount  | fraud=0 medyan | fraud=1 medyan
               |        ~100    |      ~10
```

Düşük UniqueIPCount → az IP üzerinden yoğun işlem → fraud sinyali.

**Zaman Analizi**

- Saat / haftanın günü kırılımında fraud oranı %0.55–%0.85 arasında dalgalı; ciddi farklılık yok.
- Aylık bazda hafif artış sezonsallığı (Eylül-Ekim).
- Çeyreklik bazda yukarıda gösterildiği üzere ciddi drift.

**İşlem Kanalı (TransactionChannel)**

- Tek değer (`Mobile`) → modelden çıkarılmıştır.

**Kategorik Kırılımlar**

```
TransactionType (3 unique):
  Fast    662,353
  Havale  170,785
  Eft      16,426

TransactionType  fraud %:
  Eft     1.0776%
  Fast    0.6838%
  Havale  0.3701%

DeviceOSName (2 unique):
  Android 502,443   fraud %: 0.9219
  IOS     347,121   fraud %: 0.2034

CustomerSegment (9 unique):
  C   307,388   0.4174%
  B   218,908   0.5244%
  D   196,422   0.4348%
  A2   55,945   0.7811%
  P    32,716   3.7260%
  A1   31,976   1.0414%
  KP    2,128   1.3628%
  T     2,065   0.3390%
  Y     2,016   1.3889%

IsFractionalAmount:
  False  751,700   0.4301%
  True    97,864   2.1509%

DayType:
  NaN (Normal gün)      819,083   0.6279%
  Resmi Tatil            21,426   0.8634%
  Yarım Gün Resmi Tatil   9,055   0.1104%
```

**Entity Özetleri**

```
Entity         | unique  | tx/entity (med/p99/max) | fraud-yapan | yalnız-fraud
AccountNumber  | 789,861 |   1 /     2 / 10,730   |   5,338     |   5,288
DeviceId       |  13,580 |  21 /   553 /  4,373   |     785     |     523
ReceiverName   | 135,758 |   2 /    66 /  5,586   |     915     |     576
SenderName     |   7,242 |  50 /   848 /  4,634   |   1,006     |     111
CustomerName   |   6,894 |  56 /   863 /  4,665   |     981     |     109
IP_Subnet      |  29,004 |   8 /   203 /  1,021   |     709     |      32
```

AccountNumber neredeyse her satırda unique → hesap-bazlı historical FE üretilmedi. DeviceId / ReceiverName / IP_Subnet için PIT-correct aggregate üretilmiştir (bkz. Bölüm 3).

### 2.4 KVKK Çekincesiyle Çıkarılan Kolonlar

Aşağıdaki kolonlar **model özelliği olarak kullanılmadı**; yalnız grouping/aggregate üretmek için kullanıldı:

```python
PII_COLUMNS    = ("ReceiverName", "SenderName", "CustomerName")
ENTITY_COLUMNS = ("AccountNumber", "DeviceId", "ReceiverName",
                  "SenderName", "CustomerName", "IP_Subnet")

DROP_FOR_FEATURES = (
    "BusinessKey", "TransactionDate",
    "AccountNumber", "DeviceId", "ReceiverName", "SenderName",
    "CustomerName", "IP_Subnet",
    "IsFraudTransaction", "DayType",
)
DEMOGRAPHIC = ["CustomerAge", "CustomerGender", "CustomerEducation", "CustomerMaritalStatus"]
```

- **ReceiverName / SenderName / CustomerName** — gerçek müşteri ismi; modele kategorik olarak girilmez. Yalnız PIT-correct aggregate (örn. `receiver_tx_count_30d`, `receiver_fraud_rate_smoothed`) üretmek için kullanılır.
- **AccountNumber / DeviceId / IP_Subnet** — kimlik kırılımı; sadece grouping key.
- **BusinessKey** — transaction ID; sinyal taşımaz.
- **TransactionDate** — ham haliyle özellik değildir; türev (`hour`, `dow`, `is_holiday`) ve PIT aggregate'lerin zaman ekseni olarak kullanılır.
- **TransactionChannel** — tek değer (`Mobile`), düşürüldü.
- **Demografi** — final model `demographic_excluded=True` ile eğitildi; `CustomerAge / Gender / Education / MaritalStatus` modele girmez.

---

## 3. Feature Engineering

Ham 26 sütundan, anlık türevler + zaman-aware PIT (point-in-time) aggregate'leri ile **55** özelliğe genişletildi. Eğitilen final modelde feature selection sonrası **30** özellik kullanıldı (Bölüm 5).

### 3.1 Anlık Türevler

```python
def add_derived(df: pd.DataFrame, daytype_nan: str = "unknown") -> pd.DataFrame:
    out = df.copy()

    if "DeviceModel" in out.columns:
        out["DeviceParentBrand"] = out["DeviceModel"].apply(_device_parent_brand).astype(str)
        out["DeviceModel"] = _bucket_high_cardinality(out["DeviceModel"], 100, "_OTHER_")

    out["amount_log"] = np.log1p(out["TransactionAmount"].clip(lower=0))

    if daytype_nan == "normal":
        out["DayType_clean"] = out["DayType"].fillna("Normal")
    else:
        out["DayType_clean"] = out["DayType"].fillna("Unknown")
    out["is_holiday"] = (~out["DayType"].isna()).astype("int8")

    out["os_l1h"] = out["DeviceOSName"].astype(str) + "_" + out["HasMobileActivationL1H"].astype(str)
    out["os_l8h"] = out["DeviceOSName"].astype(str) + "_" + out["HasMobileActivationL8H"].astype(str)
    return out


def _device_parent_brand(model: str) -> str:
    if not isinstance(model, str): return "Unknown"
    m = model.lower()
    if "iphone" in m: return "iPhone"
    if "samsung" in m or m.startswith("sm-"): return "Samsung"
    if "redmi" in m or "xiaomi" in m or m.startswith("mi "): return "Xiaomi"
    if "huawei" in m or m.startswith("hua"): return "Huawei"
    if "oppo" in m:    return "Oppo"
    if "vivo" in m:    return "Vivo"
    if "realme" in m:  return "Realme"
    if "google" in m or "pixel" in m: return "Google"
    if "oneplus" in m: return "OnePlus"
    if "lg" in m:      return "LG"
    if "honor" in m:   return "Honor"
    if "tcl" in m:     return "TCL"
    return "Other"
```

Eklenen kolonlar:

- `amount_log` — `log1p(TransactionAmount)`, sağa çarpık dağılım normalizasyonu
- `DayType_clean` — NaN → "Unknown"
- `is_holiday` — DayType notna binary flag
- `os_l1h`, `os_l8h` — DeviceOSName × HasMobileActivationL{1,8}H crosses (sinyal interaction)
- `DeviceParentBrand` — DeviceModel string'inden parent brand
- `DeviceModel` — top-100 + `_OTHER_` bucket'a indirgendi

### 3.2 Historical PIT-Correct Aggregate'ler (Label-FREE)

Tüm aggregate'ler her satır için **strictly o satırın TransactionDate'inden ÖNCEKİ** veriden hesaplanır; current row hariç. Data leakage'ı engellemek için sıralı veri loader tarafından garanti edilir.

```python
def add_label_free_aggregates(df, device_windows=(1,7,30),
                               receiver_windows=(7,30),
                               subnet_windows=(7,)) -> pd.DataFrame:
    out = df.copy()
    times = out["TransactionDate"].to_numpy().astype("datetime64[ns]")

    dev = out["DeviceId"].to_numpy()
    out["device_tx_count_all"] = _expanding_count_prev(dev)
    for w in device_windows:
        out[f"device_tx_count_{w}d"] = _rolling_count_prev(times, dev, w)
    out["device_distinct_accounts_all"]  = _distinct_count_prev(dev, out["AccountNumber"].to_numpy())
    out["device_distinct_receivers_all"] = _distinct_count_prev(dev, out["ReceiverName"].to_numpy())
    out["device_first_seen_days_ago"]    = _first_seen_days_ago(times, dev)

    rec = out["ReceiverName"].to_numpy()
    out["receiver_tx_count_all"] = _expanding_count_prev(rec)
    for w in receiver_windows:
        out[f"receiver_tx_count_{w}d"] = _rolling_count_prev(times, rec, w)
    out["receiver_distinct_senders_all"] = _distinct_count_prev(rec, out["SenderName"].to_numpy())
    out["receiver_distinct_devices_all"] = _distinct_count_prev(rec, dev)
    out["receiver_first_seen_days_ago"]  = _first_seen_days_ago(times, rec)

    sub = out["IP_Subnet"].to_numpy()
    out["subnet_tx_count_all"]      = _expanding_count_prev(sub)
    for w in subnet_windows:
        out[f"subnet_tx_count_{w}d"]  = _rolling_count_prev(times, sub, w)
    out["subnet_distinct_devices_all"] = _distinct_count_prev(sub, dev)

    acc = out["AccountNumber"].to_numpy()
    out["account_tx_count_all"]         = _expanding_count_prev(acc)
    out["account_first_seen_days_ago"]  = _first_seen_days_ago(times, acc)

    out["device_is_first_seen"]   = (out["device_tx_count_all"]   == 0).astype("int8")
    out["receiver_is_first_seen"] = (out["receiver_tx_count_all"] == 0).astype("int8")
    out["account_is_first_seen"]  = (out["account_tx_count_all"]  == 0).astype("int8")
    return out
```

PIT-correct rolling count yardımcısı (current row hariç, son N gün):

```python
def _rolling_count_prev(times, by_vals, window_days):
    out = np.zeros(len(by_vals), dtype=np.int32)
    window = np.timedelta64(window_days, "D")
    for v, idx in _group_indices(by_vals).items():
        t  = times[idx]
        lo = np.searchsorted(t, t - window, side="left")
        hi = np.searchsorted(t, t,           side="left")  # current dahil DEĞİL
        out[idx] = hi - lo
    return out
```

Amount aggregations (IEEE-CIS UID-aggregation paterni):

```python
def _amount_aggregates_prev(by_vals, amount):
    """Per-group, current row HARİÇ, expanding mean/std/max of amount."""
    n = len(by_vals)
    mean_arr = np.zeros(n, dtype=np.float32)
    std_arr  = np.zeros(n, dtype=np.float32)
    max_arr  = np.zeros(n, dtype=np.float32)
    for v, idx in _group_indices(by_vals).items():
        a = amount[idx]
        cum_sum = np.concatenate([[0.0], np.cumsum(a)])
        cum_sq  = np.concatenate([[0.0], np.cumsum(a * a)])
        k = np.arange(len(idx))
        mean_g = np.where(k > 0, cum_sum[k] / np.maximum(k, 1), 0.0)
        var_g  = np.where(k > 0, cum_sq[k] / np.maximum(k, 1) - mean_g * mean_g, 0.0)
        std_g  = np.sqrt(np.maximum(var_g, 0.0))
        running_max = -np.inf
        max_g = np.zeros(len(idx), dtype=np.float32)
        for i in range(len(idx)):
            max_g[i] = running_max if running_max > -np.inf else 0.0
            if a[i] > running_max:
                running_max = a[i]
        mean_arr[idx] = mean_g.astype(np.float32)
        std_arr[idx]  = std_g.astype(np.float32)
        max_arr[idx]  = max_g
    return {"mean": mean_arr, "std": std_arr, "max": max_arr}

# DeviceId ve ReceiverName için amount aggregations:
for col, prefix in (("DeviceId","device"), ("ReceiverName","receiver")):
    agg = _amount_aggregates_prev(out[col].to_numpy(), amt)
    out[f"{prefix}_amount_mean_prev"]     = agg["mean"]
    out[f"{prefix}_amount_std_prev"]      = agg["std"]
    out[f"{prefix}_amount_max_prev"]      = agg["max"]
    out[f"{prefix}_amount_ratio_to_mean"] = np.where(agg["mean"] > 0, amt/agg["mean"], 1.0)
```

### 3.3 Label-DEPENDENT Aggregate'ler (Lag Uygulanır)

Etiket bilgisi gerektiren feature'lar için, "etiketin operasyonel olarak gecikmeli geldiği" varsayımı modellenir. Lag = 7 gün, Bayesian smoothing (prior=50).

```python
def add_label_dependent_aggregates(df, label_lag_days=7, prior_strength=50.0):
    out = df.copy()
    times  = out["TransactionDate"].to_numpy().astype("datetime64[ns]")
    labels = out["IsFraudTransaction"].to_numpy().astype(np.int8)
    p_global = float(labels.mean())

    for col, prefix in (("DeviceId","device"), ("ReceiverName","receiver")):
        rate, n = _smoothed_fraud_rate_with_lag(
            times, out[col].to_numpy(), labels,
            label_lag_days, prior_strength, p_global,
        )
        out[f"{prefix}_fraud_rate_smoothed"] = rate   # (k + m*p_global) / (n + m)
        out[f"{prefix}_label_n"] = n
    return out


def _smoothed_fraud_rate_with_lag(times, by_vals, labels, lag_days, prior_strength, p_global):
    rate = np.full(len(by_vals), p_global, dtype=np.float32)
    n_arr = np.zeros(len(by_vals), dtype=np.int32)
    lag = np.timedelta64(lag_days, "D")
    for v, idx in _group_indices(by_vals).items():
        t = times[idx]; y = labels[idx]
        cum_y = np.concatenate([[0], np.cumsum(y)])
        cum_n = np.arange(len(idx) + 1)
        upper = np.searchsorted(t, t - lag, side="left")
        n = cum_n[upper]; k = cum_y[upper]
        rate[idx]  = (k + prior_strength * p_global) / (n + prior_strength)
        n_arr[idx] = n
    return rate, n_arr
```

### 3.4 Eklenen Tüm Feature'lar (Özet)

```
Device:    device_tx_count_all, device_tx_count_{1,7,30}d,
           device_distinct_accounts_all, device_distinct_receivers_all,
           device_first_seen_days_ago, device_is_first_seen,
           device_amount_mean_prev, device_amount_std_prev,
           device_amount_max_prev, device_amount_ratio_to_mean,
           device_fraud_rate_smoothed, device_label_n

Receiver:  receiver_tx_count_all, receiver_tx_count_{7,30}d,
           receiver_distinct_senders_all, receiver_distinct_devices_all,
           receiver_first_seen_days_ago, receiver_is_first_seen,
           receiver_amount_mean_prev, receiver_amount_std_prev,
           receiver_amount_max_prev, receiver_amount_ratio_to_mean,
           receiver_fraud_rate_smoothed, receiver_label_n

Subnet:    subnet_tx_count_all, subnet_tx_count_7d, subnet_distinct_devices_all

Account:   account_tx_count_all, account_first_seen_days_ago, account_is_first_seen

Instant:   amount_log, DayType_clean, is_holiday,
           os_l1h, os_l8h, DeviceParentBrand
```

### 3.5 Cleaning

- **Sıralama:** Loader, frame'i `TransactionDate` (mergesort, stable) artan sıralar. Tüm rolling/expanding aggregate'ler bu sıraya bağımlıdır.
- **NaN politikası:** `DayType` %96 NaN → "Unknown" kategorisi ile doldurulur (`is_holiday` flag ayrıca tutulur). Diğer NaN'lar preprocessor içinde `SimpleImputer(strategy="median"/"most_frequent")` ile train üzerinden öğrenilen değerlerle doldurulur.
- **PIT current-row exclusion:** Tüm aggregate'ler current row'u dışlar; ilk gözlem için historical değer 0 / global ortalamadır.
- **Yüksek kardinalite:** `DeviceModel` 1,004 unique → top-100 + `_OTHER_` bucket'lanır (HGB 255 kardinalite limiti için).

---

## 4. Train / Val / Test Split

### 4.1 Time-Based Split

Modelin gerçekçi başarısını ölçmek için rastgele değil, zamana göre sıralı 3 parça kullanıldı:

```yaml
# configs/split.yaml
split:
  strategy: time_based
  train_end:  "2024-03-31 23:59:59"
  val_start:  "2024-04-01 00:00:00"
  val_end:    "2024-05-31 23:59:59"
  test_start: "2024-06-01 00:00:00"
  test_end:   "2024-09-30 23:59:59"
```

```python
def time_based_split(df: pd.DataFrame) -> SplitResult:
    cfg = _cfg()["split"]
    train_end  = pd.Timestamp(cfg["train_end"])
    val_start  = pd.Timestamp(cfg["val_start"]);  val_end  = pd.Timestamp(cfg["val_end"])
    test_start = pd.Timestamp(cfg["test_start"]); test_end = pd.Timestamp(cfg["test_end"])
    train = df[df["TransactionDate"] <= train_end].copy()
    val   = df[(df["TransactionDate"] >= val_start)  & (df["TransactionDate"] <= val_end)].copy()
    test  = df[(df["TransactionDate"] >= test_start) & (df["TransactionDate"] <= test_end)].copy()
    return SplitResult(train, val, test, "time_based", {...})
```

### 4.2 Leakage Control

```python
def assert_no_future_leakage(train_df, val_df, date_col="TransactionDate"):
    if not (train_df[date_col].max() < val_df[date_col].min()):
        raise AssertionError(
            f"Future leakage: train max ({train_df[date_col].max()}) "
            f">= val min ({val_df[date_col].min()})."
        )
```

Çıktı:

```
Date Range:
2022-09-17 00:13:54.807000 -> 2024-09-13 23:55:51.807635

Total Rows: 849,564
Chronologically Sorted: True

Temporal Boundaries:
Train  : 2022-09-17 → 2024-03-31
Val    : 2024-04-01 → 2024-05-31
Test   : 2024-06-01 → 2024-09-30

Temporal leakage checks passed.

DATASET SHAPES
Train Shape:      (594,694 rows)
Validation Shape: (   ~85,668 rows)
Test Shape:       (  ~158,587 rows)

FRAUD DISTRIBUTION
Train Fraud Rate: 0.7910%
Validation Fraud Rate: 0.3806%
Test Fraud Rate: 0.1502%

Fraud Counts:
Train Fraud Count: 4,704
Validation Fraud Count: 326
Test Fraud Count: 238
```

Not: Fraud oranı zaman içinde düştüğü için val ve test'te oran train'den belirgin düşüktür — bu beklenen drift'tir, test sadece raporlama için kullanılır.

### 4.3 Entity Overlap

```python
def entity_overlap(a, b) -> dict:
    out = {}
    for col in ("AccountNumber", "DeviceId", "ReceiverName", "IP_Subnet"):
        sa, sb = set(a[col].unique()), set(b[col].unique())
        inter = sa & sb
        out[col] = {
            "n_unique_a": len(sa), "n_unique_b": len(sb),
            "n_overlap": len(inter),
            "pct_of_b_seen_in_a": round(len(inter)/max(len(sb),1)*100, 2),
        }
    return out
```

Çıktı (train vs test):

```
AccountNumber  pct_of_test_seen_in_train: ~0.0%   (her satır neredeyse unique)
DeviceId       pct_of_test_seen_in_train: ~92%    (cihazlar tekrarlanır)
ReceiverName   pct_of_test_seen_in_train: ~36%
IP_Subnet      pct_of_test_seen_in_train: ~67%
```

---

## 5. Cleaning and Encoding

### 5.1 Train-Only Imputation

Eksik kategorik / sayısal değerler **yalnız train setinden** öğrenilen değerlerle doldurulur:

```python
num_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),         # median fit on train only
    ("sc",  StandardScaler(with_mean=False)),          # sparse-friendly
])
cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),  # mode fit on train only
    ("oh",  OneHotEncoder(handle_unknown="ignore",     # bilinmeyen kategori → 0 vektör
                          min_frequency=20,
                          sparse_output=True)),
])
preprocessor = ColumnTransformer([("num", num_pipe, num),
                                  ("cat", cat_pipe, cat)])
```

### 5.2 Encoding Stratejileri

| Model | Kategorik Encoding |
|---|---|
| Logistic Regression | OneHotEncoder, `handle_unknown="ignore"`, `min_frequency=20` |
| Random Forest | OneHotEncoder, aynı |
| HistGradientBoosting | OrdinalEncoder + `categorical_features=cat_idx` (native split) |
| CatBoost | Native cat_features, string + "NA" fill (`CatBoostPrep`) |

```python
# HGB için ordinal + native categorical
pre_hgb = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]), cat),
])

# CatBoost için string + native
class CatBoostPrep(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self._num_med = X[self.num_cols].median(numeric_only=True); return self
    def transform(self, X):
        out = X.copy()
        out[self.num_cols] = out[self.num_cols].fillna(self._num_med)
        for c in self.cat_cols:
            out[c] = out[c].astype("string").fillna("NA")
        return out[self.num_cols + self.cat_cols]
```

### 5.3 Nihai Kontroller

```
Remaining nulls in features:  0
Final Shapes (after FE + selection):
  Train (594,694, 30)
  Val   (~85,668, 30)
  Test  (~158,587, 30)

Encoded columns:
  ['TransactionType', 'DeviceOSName', 'CustomerSegment', 'CustomerProfession',
   'DayType_clean', 'os_l1h', 'os_l8h', 'DeviceModel', 'DeviceParentBrand']
```

---

## 6. Model Training

Bu aşamada **5 model** (1 kural-bazlı + 4 ML) eğitilmiş ve karşılaştırılmıştır:

1. Rule-based baseline (operasyonel benchmark)
2. Logistic Regression
3. Random Forest
4. HistGradientBoosting (sklearn)
5. CatBoost

Class imbalance için: LR/RF/HGB → `class_weight="balanced"` (veya `balanced_subsample`), CatBoost → `auto_class_weights="Balanced"`. Bu sayede modeller nadir görülen fraud sınıfına daha fazla odaklanır.

Hiperparametre seçimi: **Optuna TPE**, 15–30 trial, objective = val PR-AUC, test setine **dokunulmadan**.

Threshold optimizasyonu: tekli 0.5 eşiği yerine **val seti üzerinden percentile-based** (top 0.1% → HIGH, top 1% → MEDIUM) ve **alerts/day** perspektifi birlikte raporlanır (bkz. Bölüm 7).

Performans metrikleri: **PR-AUC, ROC-AUC, Precision@k, Recall@k, F1@1%, F2@1%, Brier**.

### 6.1 Rule-Based Baseline

```python
def _rule_based_predict(df) -> np.ndarray:
    s = np.zeros(len(df), dtype=np.float32)
    s += 0.40 * (df["HasMobileActivationL1H"] == 1).to_numpy()
    s += 0.20 * (df["HasMobileActivationL8H"] == 1).to_numpy()
    s += 0.15 * (df["DeviceOSName"] == "Android").to_numpy()
    s += 0.10 * (df["IsFractionalAmount"] == True).to_numpy()
    s += 0.10 * (df["TransactionType"] == "Eft").to_numpy()
    s += 0.05 * (df["CustomerSegment"].isin(["P","Y","KP","A1"])).to_numpy()
    s += 0.05 * np.clip(np.log1p(df["TransactionAmount"]) / 12, 0, 1).to_numpy()
    return np.clip(s, 0, 1)
```

### 6.2 ML Modelleri ve Parametreleri

```python
"LogisticRegression": LogisticRegression(
    penalty="l2",
    C=0.0010550558461247964,         # Optuna best
    solver="liblinear",
    max_iter=500,
    class_weight="balanced",
    random_state=42,
),

"RandomForest": RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=50,
    max_features="sqrt",
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42,
),

"HistGradientBoosting": HistGradientBoostingClassifier(
    learning_rate=0.0412009387,      # Optuna best
    max_iter=600,
    max_leaf_nodes=23,
    min_samples_leaf=45,
    l2_regularization=1.0744927,
    max_features=0.6202487,
    class_weight="balanced",
    categorical_features=cat_indices,
    random_state=42,
),

"CatBoost": CatBoostClassifier(
    iterations=600,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3.0,
    bagging_temperature=1.0,
    auto_class_weights="Balanced",
    cat_features=cat_indices,
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
),
```

### 6.3 Final Model — Kullanılan 30 Kolon

Permutation importance bazlı seçilen 30 feature (test setine dokunulmadan, train+val üzerinden seçildi):

```
device_fraud_rate_smoothed, receiver_fraud_rate_smoothed, DeviceModel,
device_first_seen_days_ago, receiver_first_seen_days_ago, CustomerProfession,
UniqueIPCount, os_l8h, CustomerTenure, IsFractionalAmount,
receiver_distinct_devices_all, DeviceParentBrand, CustomerSegment,
receiver_amount_mean_prev, receiver_amount_max_prev, receiver_distinct_senders_all,
account_tx_count_all, HasMobileActivationL8H, device_is_first_seen,
receiver_label_n, receiver_tx_count_all, device_tx_count_7d,
receiver_is_first_seen, device_tx_count_1d, device_amount_ratio_to_mean,
amount_log, DayType_clean, device_amount_mean_prev, HasMobileActivationL1H,
account_is_first_seen
```

Demografi (`CustomerAge`, `CustomerGender`, `CustomerEducation`, `CustomerMaritalStatus`) modele girmez.

### 6.4 Model Karşılaştırma Sonuçları

4 model × 4 feature set eğitildi (full_safe / selected_by_permutation / drift_robust / label_free). Val PR-AUC ile sıralı:

```
Model     | Params   | Feature set              | n_feat | Val PR-AUC | Test PR-AUC | Test R@0.5% | Test R@1% | Test P@1% | Test Brier
----------|----------|--------------------------|--------|------------|-------------|-------------|-----------|-----------|----------
catboost  | baseline | selected_by_permutation  |   30   |   0.9621   |   0.8417    |   0.8992    |  0.9244   |  0.1388   |  0.00096
catboost  | baseline | full_safe                |   50   |   0.9613   |   0.8544    |   0.9034    |  0.9202   |  0.1382   |  0.00098
hgb       | tuned    | full_safe                |   50   |   0.9532   |   0.8144    |   0.8739    |  0.9034   |  0.1356   |  0.00201
hgb       | tuned    | selected_by_permutation  |   30   |   0.9498   |   0.7943    |   0.8739    |  0.8908   |  0.1338   |  0.00216
rf        | baseline | selected_by_permutation  |   30   |   0.9414   |   0.7972    |   0.8950    |  0.9160   |  0.1375   |  0.00818
hgb       | baseline | full_safe                |   50   |   0.9245   |   0.7664    |   0.8613    |  0.9076   |  0.1363   |  0.00165
rf        | baseline | full_safe                |   50   |   0.8974   |   0.7067    |   0.8403    |  0.8992   |  0.1350   |  0.00663
catboost  | baseline | drift_robust             |   41   |   0.8652   |   0.7046    |   0.7983    |  0.8403   |  0.1262   |  0.00249
hgb       | tuned    | drift_robust             |   41   |   0.8536   |   0.6486    |   0.7941    |  0.8613   |  0.1293   |  0.00543
lr        | tuned    | full_safe                |   50   |   0.7874   |   0.3837    |   0.8193    |  0.8782   |  0.1319   |  0.00607
lr        | baseline | full_safe                |   50   |   0.7696   |   0.3545    |   0.7941    |  0.8866   |  0.1331   |  0.00681
rf        | baseline | drift_robust             |   41   |   0.7531   |   0.5619    |   0.7143    |  0.7689   |  0.1155   |  0.01757
catboost  | baseline | label_free               |   46   |   0.6887   |   0.4497    |   0.6513    |  0.6933   |  0.1041   |  0.00383
hgb       | tuned    | label_free               |   46   |   0.6251   |   0.4045    |   0.6345    |  0.7605   |  0.1142   |  0.01062
rf        | baseline | label_free               |   46   |   0.4903   |   0.2396    |   0.5042    |  0.5924   |  0.0890   |  0.01985
lr        | tuned    | label_free               |   46   |   0.3872   |   0.1709    |   0.5588    |  0.6597   |  0.0991   |  0.01213
```

**Final aday: catboost / baseline / selected_by_permutation** (val PR-AUC 0.9621, 30 feature).

Final model val + test metrikleri:

```
                          Validation    Test
PR-AUC                     0.9621      0.8417
ROC-AUC                    0.9994      0.9919
n_positives                  326         238
fraud_rate                 0.3806%     0.1502%
Brier (uncalibrated)       0.00089     0.00096

Precision @ top-0.1%       1.0000      0.9747
Recall    @ top-0.1%       0.2638      0.6471
Capture   @ top-0.1%      26.38%      64.71%

Precision @ top-0.5%       0.7266      0.2702
Recall    @ top-0.5%       0.9540      0.8992
Capture   @ top-0.5%      95.40%      89.92%

Precision @ top-1%         0.3711      0.1388
Recall    @ top-1%         0.9755      0.9244
Capture   @ top-1%        97.55%      92.44%

Precision @ top-5%         0.0759      0.0290
Recall    @ top-5%         0.9969      0.9664
F1        @ top-1%         0.5376      0.2414
F2        @ top-1%         0.7358      0.4336
```

### 6.5 Artifact Yönetimi

Her aday joblib ile kaydedilir; final aday için ayrıca feature listesi + threshold + metadata yazılır:

```
artifacts/models/
├── final_model.joblib                              ← CatBoost / selected_by_permutation
├── final_catboost__full_safe.joblib
├── final_catboost__drift_robust.joblib
├── final_catboost__label_free.joblib
├── final_hgb__{full_safe, selected_by_permutation, drift_robust, label_free}.joblib
├── final_rf__{...}.joblib
├── final_lr__{...}.joblib
├── feature_list.json     ← final model'in 30 feature'ı
├── thresholds.json       ← {HIGH, MEDIUM} cut-off'ları
└── metadata.json         ← model tipi, params, tarih aralıkları, metrikler
```

`metadata.json` içeriği:

```json
{
  "model_name": "final_model",
  "model_type": "CatBoostClassifier (auto_class_weights=Balanced)",
  "feature_set_name": "selected_by_permutation",
  "n_features": 30,
  "demographic_excluded": true,
  "params": {
    "iterations": 600, "learning_rate": 0.05, "depth": 8,
    "l2_leaf_reg": 3.0, "bagging_temperature": 1.0,
    "auto_class_weights": "Balanced"
  },
  "train_date_range":      {"start": "2022-09-17", "end": "2024-03-31"},
  "validation_date_range": {"start": "2024-04-01", "end": "2024-05-31"},
  "test_date_range":       {"start": "2024-06-01", "end": "2024-09-30"},
  "label_dependent_features_enabled": true,
  "label_availability_lag_days": 7,
  "random_seed": 42
}
```

---

## 7. Threshold, Calibration ve Explainability

### 7.1 Threshold Politikası

Skor bir olasılık değeridir; karar **iş kapasitesi** ile eşleştirilen eşik üzerinden verilir.

```yaml
# configs/thresholds.yaml
policy:
  bands:
    HIGH:   {percentile: 0.001, action: BLOCK_OR_STEP_UP}   # top 0.1%
    MEDIUM: {percentile: 0.01,  action: MANUAL_REVIEW}      # top 1%
    LOW:    {percentile: 1.0,   action: ALLOW}

dataset_volume:
  total_transactions: 849564
  total_days: 727
  tx_per_day_avg: 1168
```

Val setinden hesaplanan cut-off'lar:

```json
{
  "HIGH":   0.9999968355278889,
  "MEDIUM": 0.0448862090269164
}
```

Global top-k (test seti):

```
k       n_alerts  threshold  Precision  Recall   F1     F2     capture%
0.001       158    0.9897     0.9747   0.6471  0.7778  0.6937   64.7%
0.0025      396    0.4264     0.5152   0.8571  0.6435  0.7567   85.7%
0.005       792    0.1177     0.2702   0.8992  0.4155  0.6135   89.9%
0.01       1585    0.0293     0.1388   0.9244  0.2414  0.4336   92.4%
0.025      3962    0.0055     0.0563   0.9370  0.1062  0.2269   93.7%
0.05       7924    0.0016     0.0290   0.9664  0.0564  0.1296   96.6%
0.10      15847    0.0005     0.0146   0.9748  0.0288  0.0691   97.5%
```

Business scenario — alerts/day (test seti):

```
alerts/day   total_alerts   threshold   Precision   Recall
   50           5,200        0.0034     0.0437     0.9538
  100          10,400        0.0010     0.0221     0.9664
  350          36,400        0.0001     0.0065     0.9874
 1000         104,000        0.0000     0.0023     1.0000
```

### 7.2 Calibration

Val'de fit, test'te değerlendirildi:

```
Calibration       Brier (↓ iyi)   Test PR-AUC
Uncalibrated         0.00096        0.8417
Platt (sigmoid)      0.00056        0.8417
Isotonic             0.00045        0.8251
```

PR-AUC kalibrasyondan etkilenmez (ranking metriği); Platt operasyonel kullanımda önerilir.

### 7.3 Explainability — Permutation Importance (Val)

```
Feature                          importance_mean   std
device_fraud_rate_smoothed         0.29773       0.01380
receiver_fraud_rate_smoothed       0.19251       0.00424
device_first_seen_days_ago         0.02605       0.00336
UniqueIPCount                      0.02027       0.00417
CustomerTenure                     0.01080       0.00147
receiver_first_seen_days_ago       0.00867       0.00244
os_l8h                             0.00626       0.00352
DeviceModel                        0.00546       0.00242
IsFractionalAmount                 0.00419       0.00013
DeviceParentBrand                  0.00380       0.00223
HasMobileActivationL8H             0.00258       0.00039
CustomerProfession                 0.00168       0.00053
CustomerSegment                    0.00122       0.00074
receiver_amount_mean_prev          0.00090       0.00055
receiver_amount_max_prev           0.00085       0.00093
```

API tarafında her request için SHAP TreeExplainer ile top-5 reason code üretilir:

```python
def _reason_codes(pipe, df, top_n: int = 5) -> list[ReasonCode]:
    import shap
    pre = pipe.named_steps["pre"]; clf = pipe.named_steps["clf"]
    X = pre.transform(df)
    if _state["explainer"] is None:
        try:    _state["explainer"] = shap.TreeExplainer(clf)
        except: _state["explainer"] = shap.LinearExplainer(clf, np.zeros((1, X.shape[1])))
    sv = _state["explainer"](X)
    vals = sv.values[0]
    if vals.ndim == 2: vals = vals[:, 1]
    order = np.argsort(-np.abs(vals))[:top_n]
    return [ReasonCode(
        feature=str(feature_names[i]),
        contribution=float(vals[i]),
        direction="increases_risk" if vals[i] > 0 else "decreases_risk",
    ) for i in order]
```

---

## 8. API

FastAPI + Pydantic ile geliştirilmiş; tek işlem skorlayan bir REST servisi.

### 8.1 Model ve Dosyaların Yüklenmesi

Model dosyası (`.joblib`), feature listesi, threshold ve preprocessor ilk request'te lazy load edilir; sonraki request'lerde process state'te tutulur.

```python
MODEL_NAME = os.environ.get("MODEL_NAME", "final_model")
MODEL_PATH = REPO_ROOT / "artifacts" / "models" / f"{MODEL_NAME}.joblib"

_state: dict = {"pipeline": None, "thresholds": None, "feature_cols": None,
                "explainer": None, "feature_names_after_pre": None}

def _load_pipeline():
    if _state["pipeline"] is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model artefaktı bulunamadı: {MODEL_PATH}.")
        _state["pipeline"]   = joblib.load(MODEL_PATH)
        _state["thresholds"] = _load_thresholds()
    return _state["pipeline"]
```

### 8.2 Esnek Veri Şeması (Pydantic)

```python
class TransactionInput(BaseModel):
    BusinessKey: str
    AccountNumber: int
    TransactionDate: datetime
    TransactionType: Literal["Fast", "Havale", "Eft"]
    ReceiverName: str
    SenderName: str
    HasMobileActivationL1H: int = Field(ge=0, le=1)
    HasMobileActivationL8H: int = Field(ge=0, le=1)
    DayType: str | None = None
    CustomerName: str
    CustomerSegment: str
    CustomerAge: int = Field(ge=0, le=120)
    CustomerTenure: float = Field(ge=0)
    CustomerEducation: str | None = None
    CustomerProfession: str | None = None
    CustomerMaritalStatus: str | None = None
    CustomerGender: str
    IsFractionalAmount: bool
    TransactionAmount: float = Field(ge=0)
    DeviceModel: str
    DeviceOSName: str
    DeviceId: str
    IP_Subnet: str
    UniqueIPCount: int = Field(ge=0)
    historical_features: dict | None = Field(
        default=None,
        description="device_tx_count_30d, receiver_tx_count_30d, ... PIT-correct."
    )


class ReasonCode(BaseModel):
    feature: str
    contribution: float
    direction: Literal["increases_risk", "decreases_risk"]


class ScoreResponse(BaseModel):
    fraud_score: float = Field(ge=0, le=1)
    risk_band: Literal["LOW", "MEDIUM", "HIGH"]
    is_fraud: bool
    threshold_used: float
    model_version: str
    score_calculated_at: datetime
    reason_codes: list[ReasonCode] | None = None
```

### 8.3 Feature Builder

API, kendisine gelen ham işlem nesnesinden anlık türevleri (`amount_log`, `is_holiday`, `os_l1h`, `os_l8h`, `DayType_clean`, `DeviceParentBrand`) çalışma anında üretir. Historical aggregate'ler client tarafından (feature store / online feature service'ten) request body içinde gönderilir; gönderilmezse eksik kolonlar `0` ile doldurulur.

```python
def _build_features_row(tx: TransactionInput) -> pd.DataFrame:
    row = tx.model_dump()
    hist = row.pop("historical_features", None) or {}
    df = pd.DataFrame([row])
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    df = add_derived(df)                  # instant türevler
    for k, v in hist.items():
        df[k] = v
    for k in _expected_features():        # final_model'in 30 kolonu
        if k not in df.columns:
            df[k] = 0
    return df
```

### 8.4 Karar Mekanizması

Model olasılık (0.0–1.0) üretir; band ataması val'den hesaplanan cut-off'larla yapılır.

```python
def predict(tx: TransactionInput, explain: bool = False) -> ScoreResponse:
    pipe = _load_pipeline()
    df = _build_features_row(tx)
    proba = float(pipe.predict_proba(df)[0, 1])

    cutoffs = _load_score_cutoffs()
    high_cut   = cutoffs.get("HIGH",   0.95)
    medium_cut = cutoffs.get("MEDIUM", 0.5)
    if   proba >= high_cut:   band = "HIGH";   threshold_used = high_cut
    elif proba >= medium_cut: band = "MEDIUM"; threshold_used = medium_cut
    else:                     band = "LOW";    threshold_used = medium_cut

    reason_codes = _reason_codes(pipe, df) if explain else None
    return ScoreResponse(
        fraud_score=proba,
        risk_band=band,
        is_fraud=band in ("HIGH", "MEDIUM"),
        threshold_used=threshold_used,
        model_version=MODEL_NAME,
        score_calculated_at=datetime.now(timezone.utc),
        reason_codes=reason_codes,
    )
```

### 8.5 Endpoint'ler

```python
app = FastAPI(title="Fraud Detection API",
              description="Para transferi sahtecilik tespiti. Tek-tx skorlama.",
              version="0.1.0")

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/score", response_model=ScoreResponse)
def score(tx: TransactionInput, explain: bool = Query(default=False)) -> ScoreResponse:
    return predict(tx, explain=explain)
```

Çalıştırma:

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

---

## 9. API Test

Test setinden hem **HIGH-risk (fraud)** hem **LOW-risk (legit)** örnekler seçilerek API JSON payload formatına dönüştürüldü ve `/score` üzerinden test edildi.

### 9.1 Health Check

**GET /health**

```json
{
  "status": "ok",
  "model": "final_model"
}
```

### 9.2 HIGH-Risk Request (explain=true)

**POST /score?explain=true**

```json
{
  "BusinessKey": "TX-2024-06-28-001",
  "AccountNumber": 78901234,
  "TransactionDate": "2024-06-28T07:40:46.679303",
  "TransactionType": "Fast",
  "ReceiverName": "RCV_56321",
  "SenderName": "SND_2042",
  "HasMobileActivationL1H": 1,
  "HasMobileActivationL8H": 1,
  "DayType": null,
  "CustomerName": "CUST_31022",
  "CustomerSegment": "B",
  "CustomerAge": 41,
  "CustomerTenure": 12.5,
  "CustomerEducation": "İlköğretim",
  "CustomerProfession": "Öğretmen",
  "CustomerMaritalStatus": "Evli",
  "CustomerGender": "Erkek",
  "IsFractionalAmount": true,
  "TransactionAmount": 7951.97,
  "DeviceModel": "iPhone 12",
  "DeviceOSName": "IOS",
  "DeviceId": "DEV_98231",
  "IP_Subnet": "10.42.0.0/24",
  "UniqueIPCount": 179,
  "historical_features": {
    "device_fraud_rate_smoothed":   0.082,
    "receiver_fraud_rate_smoothed": 0.041,
    "device_first_seen_days_ago":   2.1,
    "receiver_first_seen_days_ago": 5.0,
    "device_tx_count_1d":   8,
    "device_tx_count_7d":   34,
    "device_amount_mean_prev": 3120.5,
    "device_amount_max_prev":  8200.0,
    "device_amount_ratio_to_mean": 2.55,
    "receiver_tx_count_all": 14,
    "receiver_distinct_devices_all": 6,
    "receiver_distinct_senders_all": 11,
    "receiver_amount_mean_prev": 4800.0,
    "receiver_amount_max_prev":  9300.0,
    "account_tx_count_all": 0,
    "account_is_first_seen": 1,
    "receiver_label_n": 12
  }
}
```

Response:

```json
{
  "fraud_score": 0.9997736737208821,
  "risk_band": "MEDIUM",
  "is_fraud": true,
  "threshold_used": 0.0448862090269164,
  "model_version": "final_model",
  "score_calculated_at": "2026-05-13T11:00:40.536384Z",
  "reason_codes": [
    {"feature": "device_fraud_rate_smoothed",   "contribution":  7.7556, "direction": "increases_risk"},
    {"feature": "receiver_fraud_rate_smoothed", "contribution":  7.6728, "direction": "increases_risk"},
    {"feature": "DeviceModel",                  "contribution":  1.8411, "direction": "increases_risk"},
    {"feature": "device_amount_ratio_to_mean",  "contribution":  1.1316, "direction": "increases_risk"},
    {"feature": "DeviceParentBrand",            "contribution": -0.7865, "direction": "decreases_risk"}
  ]
}
```

### 9.3 LOW-Risk Request

**POST /score**

```json
{
  "BusinessKey": "TX-2024-09-11-022",
  "AccountNumber": 11223344,
  "TransactionDate": "2024-09-11T03:18:49.910035",
  "TransactionType": "Fast",
  "ReceiverName": "RCV_99812",
  "SenderName": "SND_3199",
  "HasMobileActivationL1H": 0,
  "HasMobileActivationL8H": 0,
  "DayType": "Yarım Gün Resmi Tatil",
  "CustomerName": "CUST_77821",
  "CustomerSegment": "D",
  "CustomerAge": 52,
  "CustomerTenure": 134.0,
  "CustomerEducation": "İlköğretim",
  "CustomerProfession": "İşçi",
  "CustomerMaritalStatus": "Bosanmis",
  "CustomerGender": "Erkek",
  "IsFractionalAmount": false,
  "TransactionAmount": 3159.65,
  "DeviceModel": "iPhone 13",
  "DeviceOSName": "IOS",
  "DeviceId": "DEV_44120",
  "IP_Subnet": "10.10.5.0/24",
  "UniqueIPCount": 82,
  "historical_features": {
    "device_fraud_rate_smoothed":   0.0008,
    "receiver_fraud_rate_smoothed": 0.0009,
    "device_first_seen_days_ago":   421.0,
    "receiver_first_seen_days_ago": 380.0,
    "device_tx_count_1d":  1,
    "device_tx_count_7d":  4,
    "device_amount_mean_prev": 3050.0,
    "device_amount_max_prev":  4100.0,
    "device_amount_ratio_to_mean": 1.04,
    "receiver_tx_count_all": 220,
    "receiver_distinct_devices_all": 1,
    "receiver_distinct_senders_all": 12,
    "receiver_amount_mean_prev": 2950.0,
    "receiver_amount_max_prev":  4900.0,
    "account_tx_count_all": 38,
    "account_is_first_seen": 0,
    "receiver_label_n": 200
  }
}
```

Response:

```json
{
  "fraud_score": 3.74988134909247e-08,
  "risk_band": "LOW",
  "is_fraud": false,
  "threshold_used": 0.0448862090269164,
  "model_version": "final_model",
  "score_calculated_at": "2026-05-13T11:00:40.547541Z",
  "reason_codes": null
}
```

### 9.4 Invalid Request (Missing Fields)

**POST /score** — `TransactionAmount` ve `DeviceModel` gönderilmedi.

Response:

```
HTTP 422 Unprocessable Entity

{
  "detail": [
    {"type": "missing", "loc": ["body", "TransactionAmount"],
     "msg": "Field required", "input": {...}},
    {"type": "missing", "loc": ["body", "DeviceModel"],
     "msg": "Field required", "input": {...}}
  ]
}
```

### 9.5 No historical_features (zero-fill)

**POST /score** — `historical_features` alanı boş.

```json
{
  "BusinessKey": "TX-2024-07-26-008",
  "AccountNumber": 55667788,
  "TransactionDate": "2024-07-26T15:41:52.652375",
  "TransactionType": "Fast",
  "ReceiverName": "RCV_22011",
  "SenderName": "SND_4001",
  "HasMobileActivationL1H": 0,
  "HasMobileActivationL8H": 0,
  "DayType": null,
  "CustomerName": "CUST_29011",
  "CustomerSegment": "C",
  "CustomerAge": 38,
  "CustomerTenure": 60.0,
  "CustomerEducation": "Lise",
  "CustomerProfession": "Okutman/Eğitmen",
  "CustomerMaritalStatus": "Evli",
  "CustomerGender": "Erkek",
  "IsFractionalAmount": false,
  "TransactionAmount": 5281.74,
  "UniqueIPCount": 132,
  "DeviceModel": "SM-A525F",
  "DeviceOSName": "Android",
  "DeviceId": "DEV_77810",
  "IP_Subnet": "10.20.7.0/24"
}
```

Response:

```json
{
  "fraud_score": 4.301416464301083e-08,
  "risk_band": "LOW",
  "is_fraud": false,
  "threshold_used": 0.0448862090269164,
  "model_version": "final_model",
  "score_calculated_at": "2026-05-13T11:00:40.557924Z",
  "reason_codes": null
}
```

### 9.6 Smoke Test Özet

```
#   Test                          Method   Path                  Expected     Actual
1   Health                        GET      /health               200          200
2   HIGH-risk + explain=true      POST     /score?explain=true   200          200 (score=0.9998, 5 reason codes)
3   LOW-risk                      POST     /score                200          200 (score≈0)
4   Invalid (missing fields)      POST     /score                422          422
5   No historical_features        POST     /score                200          200 (zero-fill OK)
```
