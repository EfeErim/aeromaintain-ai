# AeroMaintain AI v4 yapılacaklar

**Tarih** 6 Kasım 2026  
**Durum** Tamamlandı. Hiçbir aday kabul ölçütlerini geçmedi, v3 referansı korundu. Demo güncellenmedi ve v3 paketinde kaldı. [v4 sonuçları](results_v4.md)

v4'ün ilk hedefi, uzun kalan ömürlerdeki sistematik düşük tahmini azaltırken kritik motorları yakalama performansını korumak. FD002 ve FD004'te gerçek RUL'u 150 döngünün üzerinde olan motorlar, toplam karesel hatanın sırasıyla %69 ve %67'sini oluşturdu. Bu grupta nominal %90 tahmin aralıklarının kapsaması %25 ve %31'de kaldı. v3, bu örnekleri hem geliştirme puanlamasında hem aralık kalibrasyonunda dışarıda bırakıyordu. [v3 protokolü](protocol_v3.md) · [Referans sonuçlar](results.md)

## P0 · Tam RUL aralığında model seçimi

- [x] `docs/protocol_v4.md` taslağında motor bazlı bölmeleri, seed'leri, kesim örnekleme kuralını, metrikleri ve kabul ölçütlerini yaz. Uzun RUL örneklerini genel puanlamaya ve kalibrasyona dahil et. Tüm adaylar için aynı motorları, kesimleri ve üst sınır uygulanmamış gerçek RUL etiketlerini kullan. RUL bantları için örnek sayısını ayrıca raporla.
- [x] FD002 ve FD004 geliştirme verisinde üç referans çalıştır. Bunlar v3 ayarlarıyla yeniden eğitilen, hedefi 125 döngüde kesilen XGBoost ile üst sınır uygulanmadan eğitilen Ridge ve XGBoost. İlk karşılaştırmada özellikleri ve koşul normalizasyonunu sabit tut. Böylece aynı XGBoost ayarlarıyla hedef sınırının etkisi görülebilir.
- [x] Üst sınır uygulanmayan adaylarda koşul normalizasyonunu açık ve kapalı çalıştır. Normalizasyonu ve diğer ön işlem adımlarını her katın eğitim motorlarında yeniden öğren. XGBoost erken durdurması için puanlanan motorlardan ayrı bir iç bölme kullan.
- [x] Her alt kümede genel RMSE, MAE ve motor başına NASA skorunu raporla. `>150` bandında RMSE ve işaretli yanlılığı, `≤30` kritik grubu için de kaçırılan motor sayısını göster. Karşılaştırma belirsizliğini motor düzeyinde eşleştirilmiş yeniden örneklemeyle hesapla. Aynı motorun farklı kesimlerini bağımsız örnek sayma.
- [x] Deney tablosunu, yapılandırmayı ve seçimin gerekçesini kaydet. Daha önce açılan C-MAPSS resmî testleri kullanılırsa sonuçları **geriye dönük değerlendirme** olarak işaretle ve geliştirme sonuçlarından ayrı göster.

**Önerilen kabul kuralı.** Aday, FD002 ve FD004'ün her ikisinde `>150` RMSE'sini düşürmeli. Genel RMSE ve motor başına NASA skoru v3 referansına göre kötüleşmemeli. Kritik motor kaçırma oranı için kabul edilebilir farkı deneyden önce sabitle. Bu ölçütleri geçen aday çıkmazsa v3 referansını koru ve hangi ölçütün karşılanmadığını kaydet.

## P1 · Tahmin aralığı ve alarm

- [x] Seçilen nokta modelini sabitle. Mevcut Mondrian yöntemini tam RUL aralığında yeniden kalibre et. Ayrı kalibrasyon motorları kullanan `split conformal` referansını aynı değerlendirme motorlarında kıyasla. Değerlendirme motorlarını model seçiminden ve kalibrasyondan ayrı tut.
- [x] Nominal %90 aralık için kapsama, ortalama genişlik, *interval score* ve alt ya da üst sınırdan taşma sayılarını raporla. Sonuçları genel toplamda, RUL bandında ve yeterli örnek olan çalışma koşullarında göster. Örnek sayısı az olan grupları işaretle. [Gneiting ve Raftery](https://stat.uw.edu/research/tech-reports/strictly-proper-scoring-rules-prediction-and-estimation)
- [x] Geliştirme motorlarının geçmişini döngü döngü oynat. Nokta tahmini ve alt sınır alarmını aynı motorlarda kıyasla. İlk kalıcı alarmın arızaya kalan süresini, kaçırılan kritik motorları ve motor başına yanlış alarm olayını ölç. Ardışık kaç döngünün kalıcı alarm sayılacağını ve tekrar eden alarmların nasıl birleştirileceğini baştan tanımla. [NASA prognostik performans ölçütleri](https://ntrs.nasa.gov/citations/20100023445)

**Tamamlanma ölçütü.** Seçilen aralık ve alarm kuralı için kapsama, genişlik, alarm ön süresi ve yanlış alarm yükü aynı değerlendirme tablosunda yer almalı. Tercih edilen kuralın hangi ödünleşime dayandığı kaydedilmeli.

## P2 · Bağımsız veri üzerinde doğrulama

- [x] Yeni veri için test açılış kaydını süreçler arası kilitleme ve atomik yazmayla güçlendir. Farklı iki model aynı holdout'u eşzamanlı açmaya çalıştığında yalnızca birinin ilerleyebildiğini test et. Yayımlanan protokol ve model kilidi özetlerini deney kaydına bağla.
- [x] N-CMAPSS DS02 üzerinde veri hazırlama pilotu yap. HDF5'ten ölçülen sensörleri ve çalışma koşullarını al, uçuş döngüsü bazında özetle ve bir Ridge referansı kur. Simülatöre ait sağlık parametrelerini, sanal sensörleri ve etiket alanlarını girdi dışında tut. [N-CMAPSS veri makalesi](https://www.mdpi.com/2306-5729/6/1/5) · [N-CMAPSS_DL örnek veri hazırlığı](https://github.com/mohyunho/N-CMAPSS_DL)
- [x] Doğrulama için yeterli motor içeren bir test grubu ayır. DS02 pilotundaki üç test motoruyla sınırlı kalmak kapsama kıyası için zayıf kanıt sağlar. Modeli, aralık yöntemini ve başarı ölçütlerini sabitleyip kilit özetini yayımladıktan sonra ayrılmış motorları değerlendir. Sonuçları motor sayısı ve belirsizlikle birlikte yayımla. Çalışma koşulları arası aktarım için ayrı deney gerekiyorsa görev tanımını önceden yaz. [CRULE aktarım deneyleri](https://github.com/tilman151/crule)

**Tamamlanma ölçütü.** Ayrılmış motorlardaki sonuçlar, çalıştırma komutları, veri sürümü ve model kilidiyle birlikte incelenebilir olmalı. N-CMAPSS bulguları simülasyon doğrulaması olarak raporlanmalı.

## Sonuçları yayımlama

- [x] `results.md`, model kartı ve README'de v3 referansı, v4 geriye dönük kıyasları ve bağımsız doğrulamayı ayrı göster. Başarısız ölçütleri de kaydet.
- [x] Demo güncellenirse kullanılan veri alt kümesini, protokolü, gözlenen kapsamayı ve alarm performansını görünür tut. Paket bütünlüğünü ve demo smoke testini doğrula.
- [x] Ruff ve mevcut pytest kapsam eşiğini çalıştır. Yeni deney komutlarını ve hangi yapılandırmanın hangi sonucu ürettiğini belgeye ekle.
