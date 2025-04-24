import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# ✅ Yeni model tanımı: Training ile aynı olmalı
class DiseaseClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DiseaseClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# 📁 Dosya yolları
MODEL_PATH = "saved_model.pth"
ENCODER_PATH = "label_encoder.pkl"
CSV_PATH = "Final_Augmented_dataset_Diseases_and_Symptoms.csv"

# 📄 Dataset yükleme
print("📄 Loading dataset...")
df = pd.read_csv(CSV_PATH)
symptom_columns = df.drop(columns=["diseases"]).columns.tolist()
print(f"✅ Found {len(symptom_columns)} symptoms in dataset.")

# 📊 Scaler hazırla
print("📊 Initializing and fitting scaler...")
scaler = StandardScaler()
scaler.fit(df[symptom_columns])
print("✅ Scaler ready.")

# 🧠 Modeli yükle
print("📦 Loading trained model...")
input_size = len(symptom_columns)
num_classes = df["diseases"].nunique()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiseaseClassifier(input_size, num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("✅ Model loaded and set to eval mode.")

# 🔠 Label encoder yükle
print("🔠 Loading label encoder...")
label_encoder = joblib.load(ENCODER_PATH)
print("✅ Label encoder ready.")

# 🔄 Semptomları vektöre dönüştür
def symptoms_to_vector(symptom_list, all_symptoms):
    print("🔄 Converting symptoms to binary vector...")
    symptom_vector = [1 if symptom in symptom_list else 0 for symptom in all_symptoms]
    return np.array(symptom_vector).reshape(1, -1)

# 🤖 Tahmin fonksiyonu
def predict_from_symptoms(symptom_names):
    print("\n🤖 Starting disease prediction process...")

    # Semptom kontrolü
    print("🔎 Validating input symptoms...")
    unknowns = [s for s in symptom_names if s not in symptom_columns]
    if unknowns:
        raise ValueError(f"❌ Unknown symptoms: {unknowns}")
    print("✅ All symptoms are valid.")

    # Binary input vektörü oluştur
    binary_input = symptoms_to_vector(symptom_names, symptom_columns)

    # Ölçekle
    print("📏 Scaling input...")
    scaled_input = scaler.transform(binary_input)

    # Tensöre çevir ve modele ver
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32).to(device)

    print("🔮 Running model inference...")
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()
        top3_indices = probs.argsort()[-3:][::-1]
        top3_probs = probs[top3_indices]
        top3_labels = label_encoder.inverse_transform(top3_indices)

        print("\n🏁 Top 3 disease predictions:")
        for i in range(3):
            print(f"{i+1}. {top3_labels[i]} ({top3_probs[i]*100:.2f}%)")

# 📋 Örnek test
if __name__ == "__main__":
    test_symptoms = ["cough", "sharp abdominal pain", "vomiting", "chills"]
    print(f"📝 Input symptoms: {test_symptoms}")
    predict_from_symptoms(test_symptoms)
