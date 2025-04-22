import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from medintel_train_v3 import DiseaseClassifier
from sklearn.preprocessing import StandardScaler

# File paths
MODEL_PATH = "saved_model.pth"
ENCODER_PATH = "label_encoder.pkl"
CSV_PATH = "Final_Augmented_dataset_Diseases_and_Symptoms.csv"  # Your dataset used during training

print("📄 Loading dataset...")
df = pd.read_csv(CSV_PATH)
symptom_columns = df.drop(columns=["diseases"]).columns.tolist()

print(f"✅ Found {len(symptom_columns)} symptoms in dataset.")

# Prepare the scaler using training data
print("📊 Initializing and fitting scaler...")
scaler = StandardScaler()
scaler.fit(df[symptom_columns])
print("✅ Scaler ready.")

# Load trained model
print("📦 Loading trained model...")
input_size = len(symptom_columns)
num_classes = df["diseases"].nunique()

model = DiseaseClassifier(input_size, num_classes)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
print("✅ Model loaded and set to eval mode.")

# Load label encoder
print("🔠 Loading label encoder...")
label_encoder = joblib.load(ENCODER_PATH)
print("✅ Label encoder ready.")

# Convert symptom names to binary input vector
def symptoms_to_vector(symptom_list, all_symptoms):
    print("🔄 Converting symptoms to binary vector...")
    symptom_vector = [1 if symptom in symptom_list else 0 for symptom in all_symptoms]
    return np.array(symptom_vector).reshape(1, -1)

# Predict function
def predict_from_symptoms(symptom_names):
    print("\n🤖 Starting disease prediction process...")

    # Check for unknown symptoms
    print("🔎 Validating input symptoms...")
    unknowns = [s for s in symptom_names if s not in symptom_columns]
    if unknowns:
        raise ValueError(f"❌ Unknown symptoms: {unknowns}")
    print("✅ All symptoms are valid.")

    # Create binary input vector
    binary_input = symptoms_to_vector(symptom_names, symptom_columns)

    # Scale the input
    print("📏 Scaling input...")
    scaled_input = scaler.transform(binary_input)

    # Convert to tensor
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32)

    # Predict
    print("🔮 Running model inference...")
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).numpy().flatten()
        top3_indices = probs.argsort()[-3:][::-1]
        top3_probs = probs[top3_indices]
        top3_labels = label_encoder.inverse_transform(top3_indices)

        print("\n🏁 Top 3 disease predictions:")
        for i in range(3):
            print(f"{i+1}. {top3_labels[i]} ({top3_probs[i]*100:.2f}%)")

# Example usage
if __name__ == "__main__":
    test_symptoms = ["cough", "sharp abdominal pain", "vomiting", "chills"]
    print(f"📝 Input symptoms: {test_symptoms}")
    predict_from_symptoms(test_symptoms)
