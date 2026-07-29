# AeroMaintain AI — Sadeleştirilmiş Uygulama Planı

## 1. Proje Tanımı

### Amaç

NASA C-MAPSS FD001 sensör geçmişinden motorların kalan faydalı ömrünü (RUL)
tahmin eden ve bu tahmini ekip, hangar, parça ve günlük operasyon kapasitesi
kısıtları altında bir bakım takvimine dönüştüren yerel bir karar destek
prototipi geliştirilecek.

Proje üç şeyi birlikte göstermelidir:

1. güvenilir ve sızıntısız RUL tahmini,
2. tahmin belirsizliği ve model açıklaması,
3. tahminden uygulanabilir bakım kararına geçiş.

Bu çalışma eğitim ve portföy amaçlıdır. Gerçek uçuş emniyeti, uçuşa elverişlilik
veya bakım onayı sistemi olarak sunulmayacaktır.

### V1 kapsamı

Dahil:

- NASA C-MAPSS `FD001`
- Python 3.11
- Constant, Ridge ve XGBoost karşılaştırması
- nominal yüzde 90 tahmin aralığı
- SHAP veya lineer katsayı açıklamaları
- sentetik bakım kaynakları
- OR-Tools CP-SAT çizelgelemesi
- dört bakım politikası
- yerel Streamlit uygulaması
- pytest ve basit GitHub Actions CI

V1 dışında:

- FD002–FD004
- LSTM/CNN/Transformer
- gerçek filo ve gerçek maliyet verisi
- FastAPI, kullanıcı hesabı, veritabanı ve public hosting
- Docker zorunluluğu
- otomatik yeniden eğitim ve canlı izleme
- planlama ufku içinde yeni sensör verisiyle yeniden tahmin

---

## 2. Teknoloji ve Depo Yapısı

### Temel bağımlılıklar

- pandas, NumPy, PyArrow
- scikit-learn
- XGBoost
- SHAP
- OR-Tools
- Plotly ve Streamlit
- Pydantic ve PyYAML
- Typer
- pytest, pytest-cov ve Ruff

`pyproject.toml` bağımlılıkların ana kaynağı olacak. Python `>=3.11,<3.12`
desteklenecek; referans ortam mevcut makinedeki Python `3.11.9` olacaktır.

Kurulum:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Hedef yapı

```text
src/aeromaintain/
  data/
  features/
  models/
  optimization/
  evaluation/
  app/
  cli.py
  config.py
configs/
docs/
notebooks/
tests/
data/raw/          # Git dışında
data/processed/    # Git dışında
artifacts/         # Git dışında
```

Notebook yalnızca keşif ve görselleştirme için kullanılacak; veri hazırlama,
metrik, model ve optimizasyon mantığı `src/aeromaintain/` içinde bulunacaktır.

### Kullanıcı komutları

```text
aeromaintain prepare
aeromaintain train
aeromaintain evaluate
aeromaintain optimize
aeromaintain pipeline
aeromaintain app --run-id ID
aeromaintain doctor
```

- `pipeline`: prepare → train → calibration → model lock → evaluate → optimize
  → report sırasını çalıştırır.
- `app`: açıkça verilen doğrulanmış run kimliğini yükler; otomatik “latest”
  seçmez.
- Komutlar var olan bir run klasörünün üzerine yazmaz.
- Her run config, seed, veri hash'i, model bilgisi ve metrikleri içeren tek bir
  `manifest.json` üretir.

---

## 3. Veri Kaynağı ve Hazırlama

### Kaynak

- NASA sayfası:
  <https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data>
- İndirme:
  <https://data.nasa.gov/docs/legacy/CMAPSSData.zip>
- ZIP boyutu: `12,425,978` bayt
- ZIP SHA-256:
  `74BEF434A34DB25C7BF72E668EA4CD52AFE5F2CF8E44367C55A82BFD91A5A34F`

FD001 doğrulama değerleri:

- 100 eğitim motoru
- 100 test motoru
- 20.631 eğitim satırı
- 13.096 test satırı
- 100 test RUL etiketi
- 26 kolon: motor, çevrim, 3 operasyon ayarı, 21 sensör

NASA sayfasında lisans belirtilmediği için arşiv ve çıkarılmış ham dosyalar
Git'e eklenmeyecek. README, NASA kaynağını ve verinin simülasyon olduğunu açıkça
belirtecek.

### `prepare` davranışı

1. NASA URL'sinden indir veya `--archive PATH` ile yerel arşiv kullan.
2. ZIP boyutunu ve SHA-256 değerini doğrula.
3. Yalnızca FD001 train, test ve RUL dosyalarını çıkar.
4. Kolon, tip, motor, satır, çevrim sırası, tekrar ve sonlu değer kontrollerini
   yap.
5. Eğitim RUL hedefini üret.
6. Development/calibration motor ayrımını kaydet.
7. Parquet ve küçük veri kalite raporu üret.

Eksik veya sonlu olmayan değerler otomatik doldurulmayacak; hazırlama hata ile
duracaktır.

### Kolonlar ve hedef

```text
unit_id, cycle,
setting_1, setting_2, setting_3,
sensor_1 ... sensor_21
```

```text
rul_true = engine_max_cycle - cycle
rul_target = min(rul_true, 125)
```

`125` sınırı erken sağlıklı dönemde sensörlerden kesin RUL çıkarılamadığı için
kullanılan modelleme varsayımıdır; fiziksel gerçek olarak anlatılmayacaktır.
Resmî test metrikleri NASA'nın orijinal, sınırlandırılmamış RUL etiketleriyle
hesaplanacaktır.

---

## 4. Deney Protokolü

### Motor bazlı ayrım

1. Eğitim motorları yaşam uzunluğu çeyrekliklerine ayrılır.
2. Seed `42` ile 20 motor calibration için ayrılır.
3. Kalan 80 motor development havuzudur.
4. Model seçimi development havuzunda
   `GroupKFold(n_splits=5, shuffle=True, random_state=42)` ile yapılır.
5. Aynı motor hiçbir zaman iki rolde bulunmaz.

EDA, özellik seçimi ve hiperparametre kararları yalnızca 80 development
motorunu kullanır. Calibration motorları yalnızca tahmin aralığı için kullanılır.
Resmî test etiketleri model seçimi tamamlanmadan okunmaz.

### Özellikler

- üç operasyon ayarı
- anlık sensör değerleri
- çevrim numarası
- 5, 10 ve 20 çevrimlik:
  - ortalama
  - standart sapma
  - minimum ve maksimum
  - doğrusal eğim
  - son değer ile ortalama farkı

Kurallar:

- yalnızca mevcut ve geçmiş çevrimler kullanılır,
- sabit kolonlar her fold'un eğitim kısmında belirlenir,
- Ridge ölçekleyicisi yalnızca fold eğitim kısmında öğrenilir,
- uzun motorların eğitimi domine etmemesi için her motorun toplam örnek ağırlığı
  eşitlenir,
- feature isimleri ve sırası model artefaktına kaydedilir.

### Modeller

1. development hedef ortalaması
2. Ridge: `alpha ∈ {0.1, 1, 10, 100}`
3. XGBoost:
   - `tree_method="hist"`
   - seed `42`
   - en fazla 1.500 ağaç
   - 75 tur early stopping
   - 12 adaylık sınırlı random search

Model sıralaması:

1. en düşük motor başına normalize NASA score,
2. eşitlikte en düşük RMSE,
3. eşitlikte daha basit model.

XGBoost, Ridge'a göre development RMSE'yi en az yüzde 5 iyileştirir ve NASA
score'u kötüleştirmezse champion olur. Aksi halde Ridge champion kalır. Bu
durum projenin başarısız olduğu anlamına gelmez.

### Metrikler

- MAE
- RMSE
- NASA asymmetric score
- signed bias
- fazla RUL tahmin oranı
- gerçek RUL `<=30` motorlarda precision, recall ve F1
- RUL bantları: `0–30`, `31–60`, `61–125`, `>125`

NASA score:

```text
d = predicted_rul - true_rul

exp(-d / 13) - 1   if d < 0
exp( d / 10) - 1   if d >= 0
```

Fazla RUL tahmini daha ağır cezalandırılır. Hesaplama float64 ve `expm1` ile
yapılır.

---

## 5. Belirsizlik, Açıklama ve Test Kilidi

### Nominal yüzde 90 aralık

20 calibration motoru, motor ID sırasına göre döngüsel olarak
`20`, `60`, `100`, `126` gerçek RUL kesitlerinde değerlendirilir. Her motor
yalnızca bir calibration hatası üretir.

```text
q = calibration mutlak hatalarının yüzde 90 finite-sample quantile değeri
interval = [max(0, prediction - q), prediction + q]
```

Bu çıktı “nominal yüzde 90 ampirik tahmin aralığı” olarak adlandırılır. Resmî
testte gerçek coverage ve ortalama aralık genişliği ayrıca raporlanır; garanti
iddiası yapılmaz.

Risk bantları `interval_low` üzerinden:

- critical: `<=30`
- elevated: `31–60`
- routine: `>60`

### Açıklama

- XGBoost champion: SHAP TreeExplainer
- Ridge champion: standartlaştırılmış katsayılar
- global önem özeti ve seçili motor için yerel açıklama
- açıklamalar model davranışıdır; fiziksel nedensellik değildir

### Test kilidi

Model seçimi ve calibration tamamlandığında `model_lock.json` üretilir:

- veri hash'i
- development/calibration motor listeleri
- config ve seed
- feature listesi
- champion model ve hiperparametreleri
- model dosyası hash'i
- calibration `q`

XGBoost modeli kendi JSON biçiminde, Ridge pipeline ise `joblib` ile saklanır.
Yalnızca proje tarafından yerel olarak üretilmiş model dosyaları yüklenir ve
yüklemeden önce manifest hash'i doğrulanır.

`evaluate`, geçerli kilit olmadan resmî test etiketlerini kullanmaz. Aynı kilit
yeniden değerlendirildiğinde aynı raporun üretilmesi beklenir.

---

## 6. Bakım Senaryosu

### Varsayılan demo

- resmî testte en düşük `interval_low` değerine sahip 20 motor
- 30 günlük planlama ufku
- seed `42`
- tüm maliyetler gerçek para değil `cost_units`

Sabit maliyet varsayımları:

- planlı bakım: `100`
- arıza sonrası acil bakım: `500`
- erken bakım: kullanılmadan bırakılan tahmini çevrim başına `1`
- düşük riskli bir işi ufuk dışına erteleme: `150`

Motor başına sentetik alanlar:

- günlük 1–4 çevrim
- 2–5 gün bakım süresi
- 2–6 teknisyen ihtiyacı
- `kit_A`, `kit_B` veya `kit_C`
- 1–2 parça

Kaynaklar:

- `bay_1` ve `bay_2`
- `team_A` ve `team_B`, ekip başına 6 teknisyen
- her parça türünden başlangıçta 4 adet
- gün 10 ve 20 başında her parça türünden 3 adet ikmal
- günlük minimum operasyon talebi:
  toplam motor çevrim kapasitesinin yüzde 80'i

Sentetik değerlerin tamamı `configs/scenario.yaml` içinde ve veri sözlüğünde
işaretlenecektir.

### Gerçek etiket izolasyonu

Optimizasyon girdisi:

- RUL tahmini
- interval
- sentetik motor ve kaynak bilgileri

Optimizasyon girdisinde gerçek RUL bulunmaz. Gerçek RUL yalnızca tamamlanmış
takvimleri geriye dönük değerlendiren ayrı evaluator tarafından kullanılır.

---

## 7. Optimizasyon

### Karar değişkenleri

- motorun bakım günü
- atanmış ekip
- atanmış hangar bölmesi
- ertelenen bakım

Bir bakım işi başladığı andan itibaren motor operasyon dışında kabul edilir ve
`duration_days` boyunca aynı ekip ile aynı hangar bölmesini kullanır.

### Güvenli son gün

```text
point_rul_cycles = floor(predicted_rul)
lower_rul_cycles = floor(interval_low)

safe_due_day =
  floor(lower_rul_cycles / cycles_per_day) - 1
```

Negatif sonuç gün 0'a çekilir. Ufuk dışındaki sonuçlar “bu ufukta zorunlu
değil” olarak işaretlenir.

### Kısıtlar

- her motor bir kez bakıma girer veya açıkça ertelenir,
- aynı ekip aynı anda kapasitesini aşamaz,
- aynı hangar bölmesinde iki bakım çakışmaz,
- parça stoku hiçbir gün negatif olmaz,
- günlük çevrim kapasitesi operasyon talebinin altına düşmez,
- bakım planlama ufku dışında bitmez.

### Amaç

İki kısa aşama kullanılır:

1. güvenlik:
   - ufuk içinde due olan ertelenmiş motor sayısı,
   - safe due day sonrasındaki gecikme günleri;
2. operasyon:
   - planlı bakım maliyeti,
   - erken bakımda kaybedilen tahmini çevrim,
   - düşük riskli erteleme cezası.

OR-Tools sonucu `OPTIMAL`, `FEASIBLE`, `INFEASIBLE`, `MODEL_INVALID` veya
`UNKNOWN` olarak saklanır. Çözüm yoksa sahte takvim gösterilmez.

### Karşılaştırılacak politikalar

1. arıza sonrası reaktif bakım
2. her 90 çevrimde sabit periyotlu bakım
3. tahmini RUL `30` altına indiğinde ilk uygun güne bakım
4. CP-SAT optimize takvim

Tüm politikalar aynı motor, ekip, hangar, parça ve operasyon kapasitesini
kullanır.

Karar metrikleri:

- plansız arıza
- toplam sentetik maliyet
- erken bakımda kaybedilen çevrim
- ertelenen bakım
- gecikme günü
- ekip ve hangar kullanımı
- operasyon kapasitesi açığı
- çözüm süresi

### Sade duyarlılık kontrolü

Yalnızca üç senaryo raporlanır:

- constrained: 1 hangar, yüzde 90 operasyon talebi
- base: 2 hangar, yüzde 80 operasyon talebi
- expanded: 3 hangar, yüzde 70 operasyon talebi

Bu karşılaştırma kapasite etkisini göstermek içindir; politika ayarlamak için
kullanılmaz.

---

## 8. Streamlit Uygulaması

Dört sayfa yeterlidir:

1. **Overview**
   - model ve veri kimliği
   - ana tahmin ve karar metrikleri
   - simülasyon uyarısı
2. **Engine Health**
   - motor risk sıralaması
   - RUL ve interval
   - sensör geçmişi ve açıklama
3. **Maintenance Schedule**
   - takvim
   - ekip, hangar ve parça kullanımı
   - solver durumu ve ertelenen işler
4. **Policy Comparison & What-if**
   - dört politika
   - constrained/base/expanded karşılaştırması
   - kapasite değiştirerek optimizasyonu tekrar çalıştırma

Uygulama:

- model eğitmez,
- doğrulanmış artefaktları yükler,
- gerçek RUL'u planlama zamanında mevcutmuş gibi göstermez,
- geçersiz what-if değerlerini solver çalışmadan reddeder,
- tabloları CSV olarak indirebilir,
- eksik artefaktı anlaşılır hata mesajıyla bildirir.

---

## 9. Fazlar

### Faz 0 — Temel ve araştırma

#### M0.1 — Proje iskeleti

Teslimatlar:

- Git deposu, `.gitignore`, paket klasörleri ve başlangıç config'leri
- `pyproject.toml`, MIT `LICENSE` ve temel CLI
- çalışan `aeromaintain doctor`

Kabul:

- temiz Python 3.11 ortamında editable kurulum tamamlanıyor
- paket import ediliyor ve `doctor` ortam durumunu gösteriyor
- raw, processed ve artifacts dizinleri Git dışında

#### M0.2 — Araştırma ve veri yönetişimi

Teslimatlar:

- `docs/research.md`
- `docs/data_card.md`
- NASA kaynağı, hash, simülasyon ve yeniden dağıtım sınırları

Kabul:

- FD001 seçimi ve yöntemlerin birincil kaynakları belgeli
- veri gerçek filo ölçümü gibi sunulmuyor
- ham verinin repoda veya release'te dağıtılmayacağı açık

#### M0.3 — Kalite altyapısı

Teslimatlar:

- Ruff, pytest, coverage ve Python 3.11 CI yapılandırması
- başlangıç smoke testleri
- kanıt kaydı için `PROJECT_STATE.md`

Faz kapısı:

- `aeromaintain doctor`
- `ruff check .`
- `ruff format --check .`
- `pytest`

### Faz 1 — Veri

#### M1.1 — Güvenli veri edinimi

Teslimatlar:

- NASA indirme, ZIP hash doğrulama ve güvenli çıkarma
- yalnızca FD001 train, test ve RUL üyelerinin seçimi

Kabul:

- değişmiş ZIP ve path traversal reddediliyor
- tekrar çalıştırma ham veriyi sessizce değiştirmiyor

#### M1.2 — Parser ve veri sözleşmesi

Teslimatlar:

- 26 kolonlu parser, tip kontrolleri ve processed tablolar
- veri kalite özeti

Kabul:

- train `20.631` satır/`100` motor
- test `13.096` satır/`100` motor
- benzersiz `(unit_id, cycle)`, sıralı çevrimler ve geçerli sayılar

#### M1.3 — RUL hedefi ve motor ayrımı

Teslimatlar:

- uncapped ve `125` capped train RUL
- seed `42` development/calibration motor listeleri ve split manifesti

Kabul:

- negatif RUL yok
- motor rolleri kesişmiyor
- ayrım aynı girdiyle tekrar üretilebiliyor
- resmî test RUL eğitim kod yolunda okunmuyor

#### M1.4 — EDA ve kalite raporu

Teslimatlar:

- sensör değişkenliği, motor ömrü, null ve aykırı değer özeti
- seçili sensör trend grafikleri ve EDA raporu

Faz kapısı:

- `aeromaintain prepare`
- veri testleri
- ikinci prepare çalışmasında aynı processed hash'leri

### Faz 2 — Tahmin

#### M2.1 — Causal feature üretimi

Teslimatlar:

- anlık, motor yaşı ve 5/10/20 çevrim rolling özellikleri
- fold içinde fit edilen preprocessing
- kaydedilmiş feature isimleri ve sırası

Kabul:

- gelecekteki veri geçmiş feature'ı değiştirmiyor
- train ve inference aynı feature sırasını kullanıyor

#### M2.2 — Baseline ve grouped CV

Teslimatlar:

- hedef ortalaması, Ridge ve ortak beş GroupKFold sonuçları
- MAE, RMSE, NASA score ve RUL bandı metrikleri

Kabul:

- aynı motor fold rollerinde karışmıyor
- modeller aynı split ve metriklerle karşılaştırılıyor

#### M2.3 — XGBoost ve champion seçimi

Teslimatlar:

- en fazla 12 XGBoost adayı ve early-stopping kayıtları
- model karşılaştırma raporu ve otomatik champion kararı

Kabul:

- sabit champion kuralı uygulanıyor
- calibration ve resmî test seçimi etkilemiyor
- karar aynı config ve seed ile tekrar üretilebiliyor

#### M2.4 — Belirsizlik, açıklama ve model kilidi

Teslimatlar:

- nominal yüzde 90 interval ve risk bantları
- SHAP veya Ridge katsayı açıklaması
- kaydedilmiş model ve `model_lock.json`

Kabul:

- motor başına tek calibration skoru
- interval sınırları sıralı
- save/load tahminleri aynı
- kilit model, veri, config, feature ve calibration hash'lerini kapsıyor

#### M2.5 — Kilitli resmî test değerlendirmesi

Teslimatlar:

- test tahminleri, metrikler ve hata analizi
- kritik RUL `<=30` precision, recall ve F1

Faz kapısı:

- geçerli model kilidi olmadan evaluate çalışmıyor
- test sonucu champion veya hiperparametreleri değiştirmiyor
- aynı kilit aynı değerlendirme raporunu üretiyor

### Faz 3 — Karar optimizasyonu

#### M3.1 — Sentetik bakım senaryosu

Teslimatlar:

- en riskli 20 motor, 30 gün ve seed `42` senaryosu
- motor, ekip, hangar, parça ve maliyet tabloları
- sentetik alan veri sözlüğü

Kabul:

- aynı seed aynı senaryoyu üretiyor
- optimizer şemasında gerçek RUL bulunmuyor

#### M3.2 — Baseline bakım politikaları

Teslimatlar:

- reaktif, sabit 90 ve tahmini RUL 30 politikaları
- ortak politika değerlendirme arayüzü

Kabul:

- üç politika aynı filo ve kaynakları kullanıyor
- tüm karar metrikleri ortak evaluator ile hesaplanıyor

#### M3.3 — CP-SAT çizelgeleme

Teslimatlar:

- gün, ekip, hangar ve erteleme kararları
- ekip, hangar, stok, operasyon ve ufuk kısıtları
- iki aşamalı amaç ve solver durum raporu

Kabul:

- elle çözülen küçük örnekle sonuç eşleşiyor
- hiçbir çözüm kaynak kapasitesini aşmıyor
- no-solution durumunda sahte takvim üretilmiyor

#### M3.4 — Politika ve kapasite karşılaştırması

Teslimatlar:

- dört politika için ortak sonuç tablosu
- base/constrained/expanded karşılaştırması
- `docs/optimization.md`

Faz kapısı:

- solver testleri ve elle doğrulanan fixture geçiyor
- infeasible/unknown ve baseline'dan kötü sonuçlar görünür
- gerçek RUL yalnızca donmuş takvimlerin retrospective evaluator'ında

### Faz 4 — Uygulama ve teslim

#### M4.1 — Streamlit karar uygulaması

Teslimatlar:

- Genel Bakış, Motor Riskleri, Bakım Takvimi ve Politika Karşılaştırması
- CSV indirme ve doğrulanan kapasite what-if kontrolleri

Kabul:

- uygulama yalnızca açıkça verilen doğrulanmış run'ı yükler
- eksik/bozuk artefakt ve geçersiz what-if anlaşılır biçimde reddedilir
- gerçek RUL planlama zamanında gösterilmez

#### M4.2 — Uçtan uca pipeline

Teslimatlar:

- prepare → train → lock → evaluate → optimize → report akışı
- run manifesti, nihai rapor ve küçük CI fixture'ı

Kabul:

- var olan run üzerine yazılmıyor
- yarım run tamamlanmış gibi işaretlenmiyor
- küçük fixture ile uçtan uca test geçiyor

#### M4.3 — Dokümantasyon ve release hazırlığı

Teslimatlar:

- nihai README
- `docs/model_card.md`
- `docs/results.md`
- mimari özeti, optimizasyon dokümanı ve ekran görüntüleri

Kabul:

- README komutları denenmiş ve kullanılan CP-SAT yapısıyla uyumlu
- ham NASA verisi veya büyük model dosyası Git'te yok
- sınırlamalar ve sentetik varsayımlar görünür

#### M4.4 — Nihai release adayı

Faz kapısı:

- `aeromaintain doctor` ve temiz ortam kurulumu geçiyor
- `aeromaintain pipeline` tamamlanıyor
- `ruff check .`, `ruff format --check .` ve coverage hedefi geçiyor
- `aeromaintain app --run-id ID` ve Streamlit smoke testi geçiyor
- manifest veri/model/config hash'lerini içeriyor
- bütün milestone'lar `PROJECT_STATE.md` içinde kanıtlarıyla tamamlanmış

---

## 10. Test Planı

Zorunlu test grupları:

### Veri

- ZIP hash ve güvenli çıkarma
- 26 kolon
- beklenen satır ve motor sayıları
- sıralı çevrimler ve benzersiz `(unit_id, cycle)`
- negatif RUL olmaması

### ML

- motor rolleri ve GroupKFold kesişimsizliği
- causal rolling feature kontrolü
- feature sırası
- NASA score formülü
- model save/load aynı tahmini üretir
- interval sınırları sıralıdır
- test değerlendirmesi model kilidi ister

### Optimizasyon

- 1–3 motorlu elle doğrulanabilir örnek
- çok günlük ekip ve hangar kullanımı
- parça stoku
- operasyon kapasitesi
- ertelenen bakım
- infeasible ve unknown durumları

### Uçtan uca

- sentetik küçük fixture ile prepare → train → optimize → report
- Streamlit uygulaması doğrulanmış run ile açılır

Kalite komutları:

```text
ruff check .
ruff format --check .
pytest --cov=src/aeromaintain
```

Çekirdek paket için yüzde 80 coverage hedeflenir. CI, Ubuntu ve Python 3.11
üzerinde bu hızlı kontrolleri çalıştırır. Gerçek NASA verisiyle tam eğitim yerel
validation raporunda çalıştırılır; CI veri indirmez.

---

## 11. Nihai Tamamlanma Tanımı

Proje ancak aşağıdakilerin tamamı sağlandığında bitmiş sayılır:

- temiz Python 3.11 ortamında kurulabiliyor,
- tek pipeline komutu veri hazırlamadan rapora kadar çalışıyor,
- NASA veri kaynağı, hash ve simülasyon niteliği belgeli,
- motor bazlı leakage testleri geçiyor,
- üç model aynı protokolle karşılaştırılmış,
- champion model test geri bildirimi olmadan seçilmiş,
- RUL, belirsizlik ve açıklama raporlanmış,
- optimizer gerçek RUL görmüyor,
- dört bakım politikası aynı kaynaklarla karşılaştırılmış,
- ekip, hangar, parça ve operasyon kapasitesi hiçbir çözümde aşılmıyor,
- Streamlit uygulaması doğrulanmış sonuçları gösteriyor,
- negatif sonuçlar gizlenmiyor,
- gerçek filoya doğrudan genelleme veya emniyet iddiası yapılmıyor,
- README komutları ve testler doğrulanmış,
- ham NASA verisi Git'e eklenmemiş.

## 12. Sabit Varsayımlar

- veri: FD001
- seed: `42`
- RUL cap: `125`
- critical eşik: `30`
- nominal interval: yüzde `90`
- demo filosu: en riskli `20` motor
- planlama ufku: `30` gün
- sabit bakım periyodu: `90` çevrim
- hangar: base senaryoda `2`
- ekip: `2 × 6` teknisyen
- operasyon talebi: base senaryoda toplam kapasitenin yüzde `80`i
- uygulama: yerel Streamlit
- kod lisansı: MIT
