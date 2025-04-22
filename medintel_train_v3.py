import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import joblib

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

CSV_PATH = 'Final_Augmented_dataset_Diseases_and_Symptoms.csv'  # Replace with your actual dataset path
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

class DiseaseClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DiseaseClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def load_and_preprocess_data():
    print("🔍 Loading dataset...")
    df = pd.read_csv(CSV_PATH)

    print("🧼 Preprocessing features and labels...")
    y = df['diseases']
    X = df.drop(columns=['diseases'])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("⚖️ Balancing dataset by oversampling minority classes...")
    df_encoded = pd.DataFrame(X)
    df_encoded['label'] = y_encoded
    max_size = df_encoded['label'].value_counts().max()
    lst = [df_encoded[df_encoded['label'] == class_index] for class_index in df_encoded['label'].unique()]
    df_balanced = pd.concat([resample(group, replace=True, n_samples=max_size, random_state=SEED) for group in lst])

    X = df_balanced.drop(columns=['label']).values
    y = df_balanced['label'].values

    print("📏 Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("✂️ Splitting into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=SEED, stratify=y)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    input_size = X.shape[1]
    num_classes = len(np.unique(y))

    print(f"✅ Data ready! Input features: {input_size}, Classes: {num_classes}")
    return train_loader, test_loader, input_size, num_classes, label_encoder

def train(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc=f"🧠 Training Epoch {epoch+1}", leave=False)
    for inputs, labels in loop:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total * 100

def main():
    print("🚀 Starting pipeline...\n")
    train_loader, test_loader, input_size, num_classes, label_encoder = load_and_preprocess_data()

    print("\n🧠 Building model...")
    model = DiseaseClassifier(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n📈 Beginning training...\n")
    for epoch in range(EPOCHS):
        loss = train(model, train_loader, criterion, optimizer, epoch)
        print(f"📊 Epoch [{epoch+1}/{EPOCHS}] - Loss: {loss:.4f}")

    print("\n🧪 Evaluating on test data...")
    accuracy = evaluate(model, test_loader)
    print(f"\n✅ Final Accuracy: {accuracy:.2f}%")

    # Save model and label encoder
    MODEL_PATH = "saved_model.pth"
    ENCODER_PATH = "label_encoder.pkl"

    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    print(f"💾 Model saved to {MODEL_PATH}")
    print(f"💾 Label encoder saved to {ENCODER_PATH}")

if __name__ == '__main__':
    main()
