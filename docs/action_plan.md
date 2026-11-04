# AeroMaintain AI Aksiyon Planı

Tarih: 2026-11-04

## Durum

2026-11-04 itibarıyla uygulandı. Sonuçlar [`results.md`](results.md), kurallar
[`protocol_v2.md`](protocol_v2.md) içinde.

| Adım | Durum | Sonuç |
|---|---|---|
| 1. Risk kuralı | Tamam | İki kural güven aralığıyla yan yana |
| 2. Protokol v2 | Tamam | Aynı metrik tanımı, iç engine bölmesi, sonuçtan önce yazılı kural |
| 3. Model seçimi | Tamam, hedef tutmadı | XGBoost kilitlendi; RMSE 18,57 (hedef 16,6); LSTM 18,39 ile geçemedi |
| 4. Yanlış alarm | Tamam | FD001 `T=20`, FD003 `T=25`; bedeller yan yana |
| 5. Aralık | Tamam, hedef tutmadı | Kapsama korunuyor; genişlik 55,4 (hedef 45); CQR geçemedi |
| 6. Belirsizlik | Tamam | 10 seed, bootstrap; lock hash farkının sebebi kanıtlandı |
| 7. FD003 | Tamam | Temiz test RMSE 15,07 (12,61–17,50), tek açılış |
| 8. Sunum | Tamam | [Release](https://github.com/EfeErim/aeromaintain-ai/releases/tag/v0.2.0) ve [demo](https://aeromaintain-ai-demo.streamlit.app/) yayında |

### Sıradan sapma

Plan 4. adımı 5. adımdan önce gösteriyordu, ama risk eşiği aralığa bağlı olduğu için
ikisi birlikte belirlendi.

## Amaç

Bu plan, durum değerlendirmesinde başarısız ya da eksik bulunan alanları
kapatmak için hazırlandı. Süreler kaba tahmin, sayısal hedefler ise öneridir.

## Dayanak bulgular

Ölçümler, temiz bir clone'da yeniden üretilen `fd001-repro` run'ından geldi.

| Alan | Bulgu | Durum |
|---|---|---|
| Tekrarlanabilirlik | Referans metrikleri birebir aynı çıktı; model lock hash'i farklı | Metriklerde başarılı |
| Nokta tahmini kuralı (`prediction <= 30`) | 12/25 kritik motor, 0 yanlış alarm | Zayıf |
| Arayüz risk kuralı (`interval_low <= 30`) | 25/25 kritik motor, 9/75 yanlış alarm, precision %73,5 | Başarılı ama dokümanlarda yok |
| Model seçimi | Geliştirme NASA skorunun %99,99'u gerçek RUL'u 125'in üstündeki satırlardan geliyor | Kusurlu |
| Tahmin doğruluğu | Test RMSE 19,62; benzer protokolde ham Ridge 16,63 | Orta-zayıf |
| Aralık | Kapsama %89, q ±31,7 döngü | Kısmen başarılı |
| Dayanıklılık | Tek seed, 20 kalibrasyon motoru, tek veri seti | Ölçülmedi |

Testi taklit eden geliştirme ölçümü (motor başına rastgele kesim, 200 tekrar):

| Model | RMSE | MAE | NASA (medyan) | Kritik recall |
|---|---:|---:|---:|---:|
| Ridge | 20,90 | 16,58 | 623 | %71,6 |
| XGBoost | 18,46 | 14,06 | 470 | %84,7 |

### Bu sayıların sınırları

- Risk kuralının %100 recall değeri 25 kritik motora dayanıyor; %95 güven
  aralığının alt sınırı yaklaşık %87.
- XGBoost ölçümünde erken durdurma yanlılığı hâlâ var, tekrarlar aynı 80 motoru
  kullanıyor ve "gerçek RUL <= 150" filtresi analiz sırasında seçildi.
- XGBoost resmi test setinde denenmedi.
- Lock hash farkının sebebi doğrulanmadı. Çalışma ortamı kaydı olası sebep, ama
  referans lock karşılaştırma için elde yok.
- Literatür kıyası yön göstericidir; protokoller birebir aynı değil.

## Değişmez kurallar

1. FD001 test etiketleri görüldü. Yeni model, eşik ve özellik kararları yalnızca
   geliştirme verisiyle verilir.
2. Yeni FD001 sonuçları ayrı bir kilitli run olarak yayınlanır ve "protokol v2,
   test önceden görülmüştü" diye etiketlenir.
3. Temiz kontrol için FD003 test seti en sonda ve yalnızca bir kez açılır. Sonuç
   kötü çıksa da yayınlanır.
4. `m4-fd001-seed42-20260729` referans sonuçları silinmez; v1 olarak kalır.

## Adımlar

### 1. Risk kuralını yayınla

- **Süre:** Birkaç saat
- **Yapılacak:** `interval_low` precision, recall ve yanlış alarm metriklerini
  evaluator çıktısına, `reference_evidence.json`, README, model kartı,
  sonuçlar sayfası ve arayüze ekle.
- **Bitti sayılması için:** İki kural yan yana raporlanıyor ve güven aralığıyla
  birlikte yazılı.

### 2. Protokol v2'ye geç

- **Süre:** 1 gün
- **Yapılacak:**
  - Geliştirmede her motoru rastgele bir döngüde kesip son satırı skorla.
  - XGBoost erken durdurmasını skorlanan fold yerine ayrı bir iç engine
    bölmesine taşı.
  - Yeni seçim kuralını, herhangi bir yeni model sonucuna bakmadan önce yaz.
- **Bitti sayılması için:** Geliştirme ve test metrikleri aynı tanımla
  hesaplanıyor, yeni kuralın testleri var.

### 3. Modeli yeniden seç

- **Süre:** 1–2 gün
- **Bağımlılık:** 2. adım
- **Yapılacak:** Sırayla XGBoost'u v2 ile değerlendir, özellikleri sadeleştir,
  üst sınırı (125/130) geliştirmede ayarla. LSTM'e yalnızca hedef tutmazsa geç.
- **Hedef:** RMSE <= yaklaşık 16,6 (ham Ridge seviyesi).
- **Bitti sayılması için:** Yeni model seçimi yalnızca geliştirme kanıtıyla
  kilitlendi.

### 4. Yanlış alarmı azalt

- **Süre:** Yarım gün
- **Bağımlılık:** 3. adım
- **Yapılacak:** Yeni modelde risk bandını geliştirme verisiyle yeniden ölç. Ayrı
  sınıflandırıcı eklenmez; mevcut kural çalışıyor.
- **Bitti sayılması için:** Kaçırma ve yanlış alarm bedeli yan yana raporlanıyor,
  eşik gerekçesi yazılı.

### 5. Aralığı daralt

- **Süre:** 1–2 gün
- **Yapılacak:** 20 motor yerine 100 eğitim motorunun tamamıyla çapraz kalibrasyon
  kullan. Genişliği RUL bandına göre değiştir (Mondrian conformal veya CQR). Hata
  yüksek RUL'da büyüyor: MAE 0–30 bandında 12,65, 125 üstünde 29,63.
- **Hedef:** Kapsama yaklaşık %90 kalırken ortalama genişlik < 45 döngü.
- **Bitti sayılması için:** Kapsama ve genişlik bant bazında raporlanıyor.

### 6. Belirsizliği ölç

- **Süre:** 1 gün
- **Yapılacak:** 5–10 seed ile tekrarla, test metriklerine bootstrap güven aralığı
  ekle, lock hash farkının kaynağını bul.
- **Bitti sayılması için:** Her ana metrik aralıkla raporlanıyor, hangi alanların
  bayt düzeyinde tekrarlandığı yazılı.

### 7. FD003 ile temiz kontrol

- **Süre:** 2–3 gün
- **Bağımlılık:** 2–6. adımlar
- **Yapılacak:** Akışı FD003'e genişlet (iki arıza modu), protokolü dondurduktan
  sonra test setini bir kez aç.
- **Bitti sayılması için:** FD003 sonucu yayında.

### 8. Sunumu tamamla

- **Süre:** 1 gün
- **Bağımlılık:** Yayınlanacak sayıların kesinleşmesi
- **Yapılacak:**
  - GitHub açıklamasını güncelle (hâlâ "CP-SAT scheduling" diyor).
  - `v0.2.0` etiketi at.
  - Streamlit demosunu kur; NASA lisansı belirsiz olduğu için veri paylaşımını
    önce netleştir.
  - README'yi yeni sonuçlarla yaz.
- **Bitti sayılması için:** Açılabilir demo linki ve release var.

## Sıra ve gerekçe

```text
1 → 2 → 3 → 4
        ↓
        5 → 6 → 7 → 8
```

- 1. adım önce gelir, çünkü projenin en iyi sonucu şu an dokümanlarda yok ve iş
  küçük.
- 2. adım model işlerinden önce gelir, çünkü metrik yanlışsa sonraki kararlar da
  yanlış zemine oturur.
- 8. adım en sonda gelir, çünkü demoda gösterilecek sayılar ancak o zaman kesinleşir.

Toplam kaba tahmin: LSTM hariç yaklaşık 8–12 iş günü.

## Kapsam dışı

- **Gerçek filo verisi ve üretim sistemi:** Elde veri yok ve projenin amacı bu
  değil; eğitim amaçlı uyarısı kalır.
- **Bakım optimizasyonu:** Gerçek maliyet verisi olmadan geri getirilmez.
- **Küçük iş:** Bozuk JSON'da `evaluate` hâlâ ham hata veriyor; 2. adımın
  değişiklikleriyle birlikte çözülebilir.
