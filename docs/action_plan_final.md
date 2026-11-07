# AeroMaintain AI kapanış planı

**Tarih** 7 Kasım 2026  
**Hedef** Projeyi tek bir son deneyle tamamlayıp v1.0.0 olarak kapatmak. Bu plandan sonra yeni deney açılmaz.

## Kısa sonuç

Proje yöntem olarak bitmeye yakın, ama hikâyesi yarım. Elimizde üç temiz test (v2 FD003, v3 FD002/FD004) ve geliştirme verisinde yapılmış bir v4 var. v3'ün asıl sorunu olan "üst sınır uzun ömürlü motorları düşük tahmin ediyor" sorusu cevapsız kaldı. Üst sınırı kaldırmak bu hatayı düşürdü ama kritik motorları daha çok kaçırdı. Temiz C-MAPSS testi de kalmadı.

Kapanış için gereken iki şey var:

- Bu ödünleşimi çözen tek bir kural denemek.
- Hiç açılmamış 35 N-CMAPSS motoruyla son ve tek bir temiz test yapmak.

Sonuç olumlu da olsa olumsuz da olsa yayımlanır ve proje kapanır.

## Literatür ve benzer projeler bizi nereye koyuyor

**Nokta doğruluğunda öncü değiliz, ve bu artık kapatılabilecek bir fark değil.** C-MAPSS literatüründe derin modeller FD002/FD004 için bizim temiz testteki 27,4 RMSE'mizden belirgin biçimde düşük değerler raporluyor. Ancak raporlanan sayılar şu seçimlere bağlı:

- RUL üst sınırı (125 ile 130 arasında değişiyor),
- normalizasyonun nerede öğrenildiği,
- pencere uzunluğu,
- seed,
- test etiketinin de sınırlanıp sınırlanmadığı.

2026'daki bir denetim çalışması, yalnızca son modelin hangi veriyle yeniden eğitildiğinin bile RMSE'yi 1 birime kadar değiştirdiğini gösteriyor. Ayrıca bu makaleler resmî testi tekrar tekrar kullanıyor. Bizim tüm C-MAPSS testlerimiz ise açıldı, bu yüzden yeni bir model mimarisini artık temiz test edemeyiz. [Gupta, normalizasyon denetimi](https://engrxiv.org/preprint/view/8145) · [Sızıntıya dayanıklı değerlendirme](https://arxiv.org/pdf/2607.16493) · [Üst sınır 125 ile 130 örnekleri](https://arxiv.org/pdf/2109.01761)

**Projenin farkı değerlendirme dürüstlüğü.** Bulduğumuz portföy projeleri (XGBoost, Streamlit, MLflow örnekleri) hiçbiri şunları birlikte yapmıyor:

- protokolü önceden dondurmak,
- test etiketini yalnızca bir kez açmak,
- holdout ledger tutmak,
- başarısız sonucu da yayımlamak.

Bu projenin vitrini bu olmalı. [MrArod](https://github.com/MrArod/Predictive-Maintenance) · [Elouanxp](https://elouanxp.github.io/projects/turbofan-rul-prediction)

**Tahmin aralıklarında literatürle aynı çizgideyiz, bir adım eksiğiz.**

- **Konformal yöntem:** C-MAPSS'te konformal aralıklar [Javanmardi ve Hüllermeier (2023)](https://arxiv.org/abs/2212.14612) ile yerleşti. Aynı çalışma, bozulma serilerinde değiştirilebilirlik varsayımının açık bir sorun olduğunu kabul ediyor. v4'te split konformalın uzun RUL'da çöktüğünü bulduk (%51 ve %64 kapsama). Mondrian ise bant bazında nominale yakın kaldı. Bu bulgu literatürdeki bu açıkla örtüşüyor.
- **Asimetrik aralıklar:** 2026'da asimetrik konformal kantil regresyon, iyimser tahmin riskini azaltmak için C-MAPSS'e uygulandı. [IJPHM 2026, risk duyarlı aralıklar](https://papers.phmsociety.org/index.php/ijphm/article/view/4724) · [Çatallı RUL, asimetrik kayıp](https://www.alphaxiv.org/abs/2605.31241.md)
- **Eksik adım:** Bizim kritik kaçırma sorunumuz tam da bu asimetri. Ancak bu yöntemi temiz test edecek C-MAPSS verisi kalmadı, bu yüzden plana alınmıyor.

**N-CMAPSS'te yöntemimiz basit ama kurallara uygun.** [CruiseBench (2026)](https://arxiv.org/pdf/2607.19380) şu noktalarda bizimle aynı tutumda:

- girdi olarak yalnızca ölçülen sensörleri ve çalışma koşullarını kullanıyor,
- sanal sensörleri ve sağlık parametrelerini dışlıyor,
- uçuş evresi seçiminin ve RUL üst sınırının sonucu değiştirdiğini gösteriyor.

Raporladıkları RMSE (TSMixer, yaklaşık 3,5) seyir evresi penceresine dayanıyor. Bizim döngü özetli Ridge'imizle (15,5) doğrudan kıyaslanamaz. [PHM 2021 Data Challenge](https://data.phmsociety.org/2021-phm-conference-data-challenge/) · [N-CMAPSS veri makalesi](https://www.mdpi.com/2306-5729/6/1/5)

## Durum analizi

| Alan | Durum | Kapanış için anlamı |
|---|---|---|
| Temiz C-MAPSS testleri | FD001–FD004 hepsi açıldı | Yeni C-MAPSS iddiası yalnızca geliştirme ya da geriye dönük olabilir |
| Üst sınır ödünleşimi | v4: uzun RUL hatası düşüyor, kritik kaçırma artıyor | Cevapsız tek ana soru, tek kuralla denenecek |
| Tahmin aralığı | Mondrian bant bazında nominale yakın, split konformal uzun RUL'da kırık | Son testte Mondrian birincil yöntem olabilir |
| N-CMAPSS | DS02–DS07 etiketleri açıldı. DS01 (10 motor), DS08a (15) ve DS08c (10) **hiç açılmadı**, DS08d bozuk | 35 motorluk son temiz test mümkün |
| N-CMAPSS kritik tespit | Ridge kritik döngülerin %48'ini kaçırıyor, ama kapılarda bu ölçüt yoktu | Son testte kritik tespit de bir kapı olmalı |
| Literatürle kıyas | 27,4 ile literatür aynı ölçekte değil | Sınırlanmış test etiketiyle geriye dönük hesap gerekli |
| Disk | 32 GB boş, iki kopya N-CMAPSS arşivi yaklaşık 31,6 GB | DS01'i çıkarmadan önce yer açılmalı |
| Demo, CI, belgeler | Demo v3'te, CI yeşil. README dört protokolü sırayla anlatıyor | Tek bir sonuç hikâyesine indirilmeli |

## Bitiş tanımı

Proje şu dördü sağlanınca kapanır:

1. v5 son testi önceden dondurulmuş kurallarla bir kez çalışmış ve yayımlanmış olur.
2. README, sonuçlar, model kartı ve veri kartı tek bir son durumu anlatır. Literatürle kıyas da yer alır.
3. `v1.0.0` etiketi ve GitHub sürümü yayımlanır. CI ve demo smoke testi yeşildir.
4. Açık kalan her fikir "Gelecek çalışma" bölümüne yazılır ve uygulanmaz.

**Durdurma kuralı.** v5 kapıları geçmezse ikinci deneme yapılmaz. Sonuç olduğu gibi yayımlanır ve proje yine kapanır. Son testte kullanılan holdout tekrar açılamaz.

## Yapılacaklar

### 1. Yer aç ve son holdout'u ayır

- [x] İç içe kopyayı sil. `nasa_archive.zip` zaten `data_set.zip`'i içeriyor, birini tutmak yeterli. Silmeden önce iç arşivin özetini `source.json` ile karşılaştır.
- [x] `DS01-005.h5`'i çıkar. DS01, DS08a ve DS08c'den yalnızca birim kimliklerini oku, `Y` dizilerine dokunma. Motor sayılarını ve kimlik çakışmalarını kaydet.

**Tamamlanma ölçütü:** 35 motor sayılmış, hiçbir etiket okunmamış ve en az 15 GB boş yer var.

**Sonuç (7 Kasım 2026):** Tamamlandı. İç arşiv, dış arşivdeki CRC32 ve boyut kaydıyla eşleşti, ardından dış kopya silindi. Boş alan 41,7 GB. Motor sayısı 35: DS01'de 10, DS08a'da 15, DS08c'de 10. Hiçbir dosyada dev ve test kimlikleri çakışmıyor. Yalnızca birim kimlikleri okundu, motor başına döngü sayısı da hesaplanmadı (ömrü açığa çıkarırdı). Kayıt: [final_holdout_reservation.json](ncmapss_locks/final_holdout_reservation.json).

### 2. Literatürle aynı ölçekte geriye dönük kıyas

- [x] Saklı v3 test tahminlerinden (`runs/fd00x-v3-clean/official_test/predictions.parquet`) gerçek RUL'u 125'te sınırlayarak RMSE ve NASA skoru hesapla. Bu adımda model ya da eğitim değişmez.
- [x] Aynı son döngü protokolünü kullanan 3–5 makaleden FD002/FD004 sonuçlarını, kullandıkları üst sınırla birlikte bir tabloya koy. `research.md`'yi bu bölümün kaynaklarıyla güncelle.

**Tamamlanma ölçütü:** Tablo "geriye dönük, test etiketleri önceden açıldı" olarak etiketli ve README'de tek cümleyle özetlenmiş.

**Sonuç (7 Kasım 2026):** Tamamlandı. Gerçek RUL 125'te sınırlanınca temiz v3 test RMSE'si FD002'de 27,36'dan 14,74'e (13,28–16,15), FD004'te 27,40'tan 15,33'e (13,69–17,05) iniyor. Bu, 2016–2018 derin modellerinden (22–30) belirgin biçimde iyi, 2024–2026 sonuçlarına (14–19) yakın. Ancak FD002'de 57, FD004'te 67 motorun gerçek ömrü 125'i aşıyor ve sınırlama tam bu motorlardaki hatayı gizliyor. Bu yüzden başlık değeri sınırsız 27,4 olarak kalıyor. Belge: [results_retrospective.md](results_retrospective.md). Betik: `scripts/retrospective_capped_metrics.py`.

### 3. Protokol v5'i yaz ve dondur

v5'in iki parçası var. İkisi de hiçbir sonuç hesaplanmadan önce yazılır ve özeti kayda geçirilir.

**3a. C-MAPSS hibrit kural (yalnızca geliştirme).**

- [ ] Kural sabit ve ayarsız olacak: üst sınırlı tahmin 115'in altındaysa onu kullan, değilse üst sınırsız normalize XGBoost tahminini kullan. Mantık şu: kritik bölgede üst sınırlı model, doygunluk bölgesinde üst sınırsız model karar verir.
- [ ] Kural, saklı v4 seçim tahminleri (`oof.parquet`) ve aynı kesimler üzerinde v4 kapılarıyla değerlendirilir. Bu yüzden yeni model eğitimi gerekmez.
- [ ] Kural v4 sonuçları görüldükten sonra tasarlandığı için sonuç yalnızca geliştirme kanıtı sayılır.

**3b. N-CMAPSS son temiz testi.**

- [ ] **Veri:** DS02–DS07'nin tüm motorlarıyla eğit. Bu motorların etiketleri açıldı ama eğitimde kullanmak serbest. DS01, DS08a ve DS08c'deki 35 motorda tek seferlik değerlendirme yap.
- [ ] **Bunun ne ölçtüğü:** Test, alt kümeler arası genelleme testidir. Uçuş sınıfı ve arıza modu farklılıkları önceden yazılır.
- [ ] **Adaylar:** Ridge ve v3 ayarlarıyla XGBoost, aynı 72 döngü özelliğiyle. Seçim yalnızca eğitim motorlarında, motor gruplu çapraz doğrulamayla yapılır.
- [ ] **Aralıklar:** Mondrian birincil yöntem (v4 bulgusuna dayanarak), split konformal ikincil. Kalibrasyon için eğitim alt kümelerinden ayrı motorlar kullanılır.
- [ ] **Önceden yazılacak kapılar:**
  - RMSE, ortalama RUL referansından kötü olmayacak.
  - Mondrian kapsaması ≥ %85 olacak.
  - Kalıcı alarm tekrarında kaçırılan kritik motor oranı ≤ %10 olacak. Kural v4 ile aynı: üç ardışık döngü, eşik 30.
  - Örneklenen kritik kaçırma oranı ayrıca raporlanır.
- [ ] **Kilit ve kayıt:** Model kilidi ve bölmeler yayımlanır, sonra ledger'a kayıt düşülür. Kod, özeti alınmış commit'ten çalıştırılır. Böylece v4'teki "koşu sırasında kod değişti" sorunu tekrarlanmaz.

**Tamamlanma ölçütü:** `docs/protocol_v5.md` ve kilitler `main`'e pushlanmış olur, etiketler henüz açılmamış olur.

### 4. v5'i çalıştır ve yayımla

- [ ] 3a ve 3b'yi çalıştır. Çalıştırmadan önce çalışma ağacının temiz olduğunu ve commit özetini kayda geçir.
- [ ] Kanıtı v4 yayımlama betiğiyle aynı yapıda yayımla. Geçen ve kalan kapıları, motor sayısını ve güven aralıklarını göster.

**Tamamlanma ölçütü:** `docs/results_v5.md` ve kanıt JSON'u yayımlanmış ve ledger'da tek kayıt var.

### 5. Belgeleri tek bir son duruma indir

- [ ] README'yi sonuçtan başlayan bir anlatıya çevir. Sıra şöyle olsun:
  1. Ne kanıtlandı: temiz test tablosu (v2, v3, v5).
  2. Ne kanıtlanmadı: üst sınır ödünleşimi, literatür ölçeğindeki fark.
  3. Yöntem: dondurulmuş protokol ve ledger.
- [ ] Protokol geçmişini ayrı bir belgeye taşı.
- [ ] Model kartına son modeli, veri kartına N-CMAPSS'i ekle. `research.md`'ye 2. adımdaki literatürü ekle.
- [ ] "Gelecek çalışma" bölümünü yaz. Uygulanmayacak fikirler şunlar:
  - asimetrik konformal kantil regresyon,
  - derin modeller,
  - CruiseBench protokolü,
  - gerçek filo verisi.

**Tamamlanma ölçütü:** Yeni bir okuyucu README'nin ilk ekranında projenin neyi kanıtladığını ve neyi kanıtlayamadığını görebiliyor.

### 6. Sürüm ve kapanış

- [ ] **CI:** `ubuntu-latest` yerine sabit bir Ubuntu sürümü kullan. GitHub, `ubuntu-latest`'in Ubuntu 26'ya geçeceği uyarısını verdi.
- [ ] **Demo:** Hibrit kural yalnızca geliştirme kanıtıyla desteklendiği için v3 paketinde kalsın. Etiketlerini ve bağlantılarını son belgelere göre güncelle. Demo smoke testini çalıştır.
- [ ] **Sürüm:** `pyproject.toml` sürümünü 1.0.0 yap. CHANGELOG'u tamamla, `v1.0.0` etiketini ve GitHub sürümünü yayımla.
- [ ] **Kapanış:** README'ye "Proje tamamlandı" notunu ve tarihini ekle.

**Tamamlanma ölçütü:** Bitiş tanımındaki dört koşul sağlanmış.

## Sıra ve gerekçe

1. **1. adım önce gelir:** Disk dolu. DS01 çıkarılmadan da holdout sayılamaz.
2. **2. adım protokolden önce gelir:** Yalnızca açılmış veriye dokunuyor ve README'nin en zayıf noktasını kapatıyor.
3. **3. ve 4. adım tek bir son deneydir:** 3. adımda hiçbir sonuç hesaplanmaz.
4. **5. ve 6. adım deney bittikten sonra gelir:** Belgeler son sonucu anlatmalı. Arada güncellemek işi iki kez yaptırır.

## Kapsam dışı

- **Derin öğrenme ile öncü sonuç peşine düşmek:** Temiz C-MAPSS testi kalmadığı için hiçbir iyileşme bağımsız olarak doğrulanamaz.
- **Asimetrik konformal aralıklar:** Kritik kaçırma sorununa doğru cevap olabilir, ama temiz test edilecek veri kalmadı. Gelecek çalışmaya yazılır.
- **DS08d'yi onarmak:** Dosya bozuk. Onarmak sonucu belirsizleştirir.
- **Gerçek filo verisi, bakım maliyeti optimizasyonu, üretim sistemi:** Önceki planlardaki gerekçeler geçerli.
