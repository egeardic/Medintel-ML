import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from tqdm import trange
import joblib

# Constants
CSV_PATH = 'Final_Augmented_dataset_Diseases_and_Symptoms.csv'
SEED = 42
EPISODES = 300
LEARNING_RATE = 0.001
GAMMA = 0.99

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📟 Using device: {device}")

# Load and preprocess data
def load_data():
    print("🔍 Loading dataset...")
    df = pd.read_csv(CSV_PATH)

    print("⚖️ Balancing dataset...")
    y = df['diseases']
    X = df.drop(columns=['diseases'])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    df_encoded = pd.DataFrame(X)
    df_encoded['label'] = y_encoded
    max_size = df_encoded['label'].value_counts().max()
    lst = [df_encoded[df_encoded['label'] == class_index] for class_index in df_encoded['label'].unique()]
    df_balanced = pd.concat([resample(group, replace=True, n_samples=max_size, random_state=SEED) for group in lst])

    X = df_balanced.drop(columns=['label']).values
    y = df_balanced['label'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, label_encoder

# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 896)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(896, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.out(x)

# Weight initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Compute discounted rewards
def compute_discounted_rewards(rewards, gamma):
    discounted = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted.insert(0, R)
    discounted = torch.tensor(discounted, dtype=torch.float32)
    std = discounted.std()
    if std > 0:
        discounted = (discounted - discounted.mean()) / (std + 1e-9)
    else:
        discounted = discounted - discounted.mean()
    return discounted

# Training loop
def train_policy_gradient(X, y, policy_net, optimizer, episodes=500):
    print("🚀 Starting training with reinforcement learning...")
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)

    for episode in trange(episodes, desc="Training"):
        log_probs = []
        rewards = []

        for i in range(len(X)):
            input_tensor = X[i]
            label = y[i]

            logits = policy_net(input_tensor)
            action_probs = torch.softmax(logits, dim=0)
            action_probs = torch.nan_to_num(action_probs, nan=1e-9, posinf=1e-9, neginf=1e-9)
            action_probs = action_probs / action_probs.sum()

            m = Categorical(action_probs)
            action = m.sample()
            log_prob = m.log_prob(action)

            reward = 1.0 if action == label else 0.0

            log_probs.append(log_prob)
            rewards.append(reward)

        # Compute loss
        discounted_rewards = compute_discounted_rewards(rewards, GAMMA)
        loss = -torch.stack(log_probs) * discounted_rewards
        loss = loss.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Main function
def main():
    X, y, label_encoder = load_data()
    input_size = X.shape[1]
    num_classes = len(np.unique(y))

    policy_net = PolicyNetwork(input_size, num_classes).to(device)
    policy_net.apply(init_weights)

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

    train_policy_gradient(X, y, policy_net, optimizer, episodes=EPISODES)

    # Save model
    torch.save(policy_net.state_dict(), "reinforce_model.pth")
    joblib.dump(label_encoder, "label_encoder.pkl")
    print("💾 Model and label encoder saved!")

if __name__ == '__main__':
    main()
