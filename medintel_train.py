import time
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Zamanlayıcıyı başlat
start_time = time.time()

# 1. Veri setini oku
df = pd.read_csv("Final_Augmented_dataset_Diseases_and_Symptoms.csv")  # kendi dosya adını yaz

# 2. Özellikleri (X) ve hedefi (y) ayır
X = df.drop('diseases', axis=1)  # Tüm semptom sütunları
y = df['diseases']              # Hastalık sınıfı

# 3. Eğitim ve test verisine ayır
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 4. Model kur (pipeline: standartlaştır + random forest)
model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])

# 5. Eğit
model.fit(X_train, y_train)

# 6. Tahmin ve değerlendirme
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Modeli kaydet
joblib.dump(model, "disease_predictor_model1000.pkl")

# Zamanlayıcıyı durdur ve süreyi yazdır
end_time = time.time()
print(f"İşlem süresi: {end_time - start_time:.2f} saniye")
