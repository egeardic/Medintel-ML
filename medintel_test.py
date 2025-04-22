import joblib
import time
import numpy as np
import pandas as pd

start_time = time.time()
# Modeli yükle
model = joblib.load("disease_predictor_model_1.pkl")
# Semptom sütunlarını al
columns = pd.read_csv("Final_Augmented_dataset_Diseases_and_Symptoms.csv").drop('diseases', axis=1).columns
# Kullanıcıdan alınan semptomlar
input_symptoms = ["cough", "sharp abdominal pain", "vomiting", "chills"]
# Giriş vektörünü hazırla
input_vector = np.zeros((1, len(columns)))
for symptom in input_symptoms:
    if symptom in columns:
        idx = list(columns).index(symptom)
        input_vector[0, idx] = 1
# Olasılıkları tahmin et
prediction_proba = model.predict_proba(input_vector)
# Sınıf isimlerini al
class_names = model.classes_
# İlk 3 tahmini ve olasılıklarını al
top3_indices = np.argsort(prediction_proba[0])[::-1][:3]  # en yüksekten düşük olasılığa sırala
top3_diseases = class_names[top3_indices]
top3_probs = prediction_proba[0][top3_indices] * 100  # yüzdelik format
# Sonuçları yazdır
print("En olası 3 tahmin:")
for disease, prob in zip(top3_diseases, top3_probs):
    print(f"- {disease}: %{prob:.2f}")

end_time = time.time()
print(f"İşlem süresi: {end_time - start_time:.2f} saniye")