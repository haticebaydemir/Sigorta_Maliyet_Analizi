Projenin kodlarına [buraya tıklayarak](https://colab.research.google.com/drive/1dh55TzMhj9H7Np_HKdINR2kwpik-nN4y#scrollTo=bn9vXfSM448j) ulaşabilirsiniz.
# Sigorta_Maliyet_Analizi

## Libraries
```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVR
```
Burada veri seti işlemleri için Pandas, görselleştirme işlemleri için Matplotlib, makine öğrenimi algoritmalarını uygulamak ve sonuçlarını değerlendirmek için Sklearn kütüphanelerini import edeceğiz.

## Data Analysis
```
dataset = pd.read_csv("insurance.csv")
df = dataset.copy()
df.head()
df.info()
df.isna().sum() # Boş data var mı kontrol edelim.
df.describe().T
plt.figure(figsize=(10,7))
df["children"].value_counts().plot.bar() # Kaç çocuk var?
df["region"].value_counts().plot.bar()
df["region"].value_counts().plot.bar()
```
Region ve children grafiklerine baktığımızda region grafiğinin birbirlerine yakın değerlerde dağılım gösterdiğini söyleyebiliriz. Bu durumda region değerinin bir parametre olmadığını söyleyebiliriz. Ama children grafiği için bu durum böyle değil.

```
df["sex"].value_counts() # Toplamda kaç erkek kaç kadın var
print(df["sex"].value_counts())
df["sex"].value_counts().plot(kind="pie", autopct='%1.1f%%',figsize=(6,6)); # Kadın erkek dağılımı
print(df["smoker"].value_counts())
df["smoker"].value_counts().plot(kind="pie", autopct='%1.1f%%',figsize=(6,6)); # Sigara içenler içmeyenler
plt.figure(figsize=(10,7))

sns.barplot(x="region", y="bmi", hue="smoker", data=df)
```
## Data Preprocessing
 ```
sns.boxplot(df.charges); # Outliers (Aykırı Değer)
sns.boxplot(df.age);
sns.boxplot(df.children);
sns.boxplot(df.bmi);

def label_encoding(column_name):# Label encoding
  label_encoder = LabelEncoder()
  df[column_name] = label_encoder.fit_transform(df[column_name]) # Label Encoder veriyi birebir sayısallaştırmaya yarar. Yani kategorik her veriye sayısal bir değer atar.
label_encoding("smoker")
label_encoding("sex")
df.head()
dataset.head()
```
```
# One-hot encoding

one_hot = pd.get_dummies(df["region"])
one_hot.head() # One-Hot Encoding, kategorik değişkenlerin ikili (binary) olarak temsil edilmesi anlamına gelmektedir. Kategorik verilerin temsilinin daha etkileyici ve kolay olmasını sağlar.
df = pd.concat([df, one_hot], axis=1)
```
```
df.drop("region", axis=1, inplace=True)
df.head()
# Artık bütün data nümerik data oldu.
```
```
df.info()
x = df.drop("charges", axis=1)
y = df["charges"]
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x[0:5]  # Tüm sayıları 0 ve 1 arasına yeniden ölçeklendirir. Değişkenler scale edilirse daha hızlı ve daha doğru sonuç alırız.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=42)
x_train.shape
y_train.shape
x_test.shape
y_test.shape
```
## Model Selection
```
# model selection
linreg = LinearRegression()
DecTree = DecisionTreeRegressor()
Rand_forest = RandomForestRegressor()
SVM = SVR()
linreg_scores = cross_val_score(linreg,
                                x_train, y_train,
                                scoring="neg_mean_squared_error",
                                cv=10)
```
Bu kod linreg (lineer regresyon modeli) kullanılarak eğitim veri seti üzerinde 10 katlı çapraz doğrulama ile modelin ortalama karesel hata (MSE) performansını değerlendirir ve her bir kat için bu hataları döndürür. Bu sayede modelinizin ne kadar iyi veya kötü performans gösterdiğini değerlendirebilir ve modelinizi daha iyi hale getirmek için gerekli iyileştirmeleri yapabiliriz.
```
linreg_scores
# Değerler lineer regresyon olarak karşımıza çıkıyor. Cross validation yapıldıktan sonra çıkan sonuç karşımıza gelmiş oldu.
# Dizi içerisindeki her bir değer veri kümesinin belirli bir bölmesi üzerinde modelimizin tahminlerinin gerçek hedef değerlerinden ne kadar sapma gösterdiğini temsil eden bir yapıda.
```
```
DecTree_scores = cross_val_score(DecTree,
                                x_train, y_train,
                                scoring="neg_mean_squared_error",
                                cv=10)
DecTree_scores

Rand_forest_scores = cross_val_score(Rand_forest,
                                x_train, y_train,
                                scoring="neg_mean_squared_error",
                                cv=10)
Rand_forest_scores

svm_scores = cross_val_score(SVM,
                                x_train, y_train,
                                scoring="neg_mean_squared_error",
                                cv=10)
svm_scores

def score_display(scores):
  scores = np.sqrt(-scores)
  print(f"""
  RMSE Scores:{scores},
  Mean: {scores.mean()},
  Standart Deviation: {scores.std()}
  """)
```
score_display fonksiyonu, hiperparametre optimizasyonu sürecinde elde edilen skorları değerlendirir ve bu skorların istatistiksel özetini sağlar. Bu, modelin performansının ne kadar değişken olduğunu ve ortalama performansını hızlı bir şekilde değerlendirmek için kullanılır.
```
score_display(linreg_scores)
score_display(DecTree_scores)
score_display(Rand_forest_scores)
score_display(svm_scores)
```
### Hyper-parameter Optimization
Hiper parametreler, bir modelin başarıya ulaşmak için optimize edilmesi gereken özelliklerdir. Bunlar, modelin öğrenme oranı, yinelemeler sayısı, aktivasyon fonksiyonları, ağırlıklar, regulasyon parametreleri, çıkış fonksiyonu gibi özelliklerdir.
```
params = {"n_estimators": [3, 10, 20, 50],
          "n_jobs": [2, 3, 4, 10]}
grid_s = GridSearchCV(Rand_forest, params, cv=5, scoring="neg_mean_squared_error")
grid_s.fit(x_train, y_train)  # En iyi performansı sağlayan seti belirler.
grid_s.best_params_ # Bu parametrelerin hangisi daha iyi?

for mean_score, params in zip((grid_s.cv_results_['mean_test_score']), grid_s.cv_results_["params"]):
  print(np.sqrt(-mean_score),"-----------", params)   # Hiperparametre optimizasyon çalışması sonuçları.
```
Parametreleri belirliyoruz ve bu parametrelere göre job sayısı ve test skorunu gözlemleyebiliyoruz. Belirli bir hiperparametre değeri kullanıyoruz ya bu hiperparametre değerini kullanarak bir modelin farklı konfigürasyonlarını değerlendirdiğiniz bir hiperparametre optimizasyon çalışması sonuçları. Yani özellikle n estimators ve n jobs parametrelerini farklı değerlerle denemiş olduk. Bu sonuçlar modelin hiparametrelrini optimize ediyor, optimize ederken de hata metriğini minimize ediyor.

### Model Evaluation
```
prediction = grid_s.best_estimator_.predict(x_test)
```
Prediction değişkeni, en iyi parametreleri kullanarak eğitilmiş modelin, x_test veri kümesi için yaptığı tahminleri içerir. Bu tahminler, modelin x_test üzerinde ne kadar iyi performans gösterdiğini değerlendirmek için kullanılabilir.
```
y_test[0:10].values

y_test[0:10].values ifadesi, test veri kümesinin ilk 10 gerçek değerini bir NumPy dizisi olarak döndürür. Bu gerçek değerler, modelin bu veri noktaları için ne kadar doğru tahminler yaptığını değerlendirmek için kullanılabilir.


comparison = pd.DataFrame({"y_test": y_test[0:10].values,
                          "Predictions": prediction[0:10]})

````
comparison DataFrame'ini kullanarak, modelin gerçek değerlerle ne kadar uyumlu tahminler yaptığını görmek ve karşılaştırmak daha kolay olur. Bu tür bir karşılaştırma, modelin performansını hızlı bir şekilde değerlendirmek için oldukça yararlıdır
```
comparison  # Tahmin edilen değer ile gerçek değerleri bu tabloda karşılatırabiliyoruz. <br/>

def regression_evaluation(preds):
  mse = mean_squared_error(y_test, preds)
  rmse = np.sqrt(mse)
  r_squared = r2_score(y_test, preds)
  mae = mean_absolute_error(y_test, preds)

  print(f" Mean Absolute Error: {mae} \n Mean Squared Error: {mse} \n Root Mean Squared Error: {rmse} \n R.Squared Value: {r_squared}")
regression_evaluation(prediction)
```
Bu kod, regresyon modelinin tahminlerini değerlendirmek için kullanılır.

Mean Absolute Error (MAE): Gerçek değerlerle tahminler arasındaki mutlak farkların ortalamasıdır. Daha küçük bir MAE, modelin daha iyi performans gösterdiği anlamına gelir.

Mean Squared Error (MSE): Gerçek değerlerle tahminler arasındaki karelerin ortalamasıdır. Daha küçük bir MSE, modelin daha iyi bir uyum sağladığı anlamına gelir.

Root Mean Squared Error (RMSE): MSE'nin karekökü alınarak elde edilen değerdir. RMSE, tahmin hatalarının ortalama büyüklüğünü verir. Daha küçük bir RMSE, modelin daha iyi bir performans gösterdiği anlamına gelir.

R-squared (R²) değeri: Bu değer, modelin veri üzerinde ne kadar iyi uyum sağladığını gösterir. R² değeri 0 ile 1 arasında bir değer alır. 1'e yakın bir R² değeri, modelin veriyi iyi açıkladığı anlamına gelir. 0'a yaklaşan bir değer ise modelin veriyi açıklamada başarısız olduğunu gösterir.
