Projenin kodlarına [buraya tıklayarak](https://colab.research.google.com/drive/1dh55TzMhj9H7Np_HKdINR2kwpik-nN4y#scrollTo=bn9vXfSM448j) ulaşabilirsiniz.
# Sigorta Maliyetleri Tahmin Projesi

Bu proje, sigorta maliyetlerini tahmin etmek amacıyla çeşitli makine öğrenimi algoritmalarının uygulanmasını içermektedir. Proje, Python programlama dili ve çeşitli kütüphaneler kullanılarak gerçekleştirilmiştir.

## Kütüphaneler
Projenin başında aşağıdaki kütüphaneler kullanılmaktadır:

**NumPy**: Matematiksel işlemler için.

**Pandas**: Veri analizi ve manipülasyonu için.

**Seaborn** ve **Matplotlib**: Veri görselleştirmeleri için.

**Scikit-learn**: Makine öğrenimi algoritmaları ve değerlendirme metrikleri için.

## Veri Seti
Proje, "insurance.csv" adlı bir veri seti kullanmaktadır. Veri seti aşağıdaki sütunlardan oluşmaktadır:

- age: Yaş
- sex: Cinsiyet
- bmi: Vücut Kütle İndeksi
- children: Çocuk sayısı
- smoker: Sigara içme durumu (Evet/Hayır)
- region: Bölge (Kuzeydoğu, Kuzeybatı, Güneydoğu, Güneybatı)
- charges: Sigorta maliyeti

## Veri Analizi
Veri setinin temel analizi aşağıdaki şekilde gerçekleştirilmiştir:

- Boş değer kontrolü

- Temel istatistiksel analiz

- Kategorik değişkenlerin dağılımının görselleştirilmesi

## Veri Ön İşleme
Veri setindeki kategorik değişkenler sayısal verilere dönüştürülmüştür:

- **Label Encoding**: smoker ve sex sütunları sayısallaştırılmıştır.
- **One-Hot Encoding**: region sütunu ikili olarak temsil edilmiştir.
- **Ölçekleme**: Tüm sayısal veriler 0-1 aralığına ölçeklenmiştir.

## Model Seçimi
Aşağıdaki makine öğrenimi algoritmaları kullanılmıştır:

- **Lineer Regresyon**
- **Karar Ağaçları**
- **Random Forest**
- **Destek Vektör Regresyonu (SVR)**
  
Her model için 10 katlı çapraz doğrulama ile ortalama karesel hata (MSE) hesaplanmıştır. Aşağıda elde edilen RMSE değerleri bulunmaktadır:

- **Lineer Regresyon**: 6133.12
- **Karar Ağaçları**: 6515.52
- **Random Forest**: 4909.87
- **Destek Vektör Regresyonu**: 12541.32
- Random Forest modeli en iyi performansı göstermiştir.

## Hiperparametre Optimizasyonu
Random Forest modeli için en iyi hiperparametreler GridSearchCV kullanılarak optimize edilmiştir. Elde edilen en iyi parametreler şunlardır:

- n_estimators: 20
- n_jobs: 2
## Model Değerlendirme
En iyi hiperparametrelerle eğitilen model, test veri kümesi üzerinde değerlendirilmiştir. Elde edilen sonuçlar:

- **Mean Absolute Error (MAE)**: 2573.14
- **Mean Squared Error (MSE)**: 21641442.40
- **Root Mean Squared Error (RMSE)**: 4652.04
- **R² Değeri**: 0.86

## Sonuç
Bu proje, sigorta maliyetlerinin tahmin edilmesinde Random Forest modelinin en iyi performansı gösterdiğini ortaya koymuştur. Elde edilen sonuçlar, sigorta şirketlerinin poliçe fiyatlandırma stratejilerinin geliştirilmesine katkıda bulunabilir.

