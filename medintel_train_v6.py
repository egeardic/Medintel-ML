import pandas as pd
import argparse
import pickle

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm


def load_data(path):
    """
    Load CSV data and return feature matrix X and labels y.
    Assumes first column is 'diseases' and remaining are symptom features.
    """
    data = pd.read_csv(path)
    # Encode disease labels
    le = LabelEncoder()
    data['label'] = le.fit_transform(data['diseases'])

    # Features and labels
    X = data.drop(['diseases', 'label'], axis=1).values
    y = data['label'].values

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, le, scaler


def make_dataloaders(X, y, batch_size=64, test_size=0.2, random_state=42):
    """
    Split data into train/test and wrap in DataLoader objects.
    Uses random split without stratification to avoid errors when classes have few samples.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class Net(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.5):
        super(Net, self).__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_size = h
        layers.append(nn.Linear(in_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in tqdm(loader, desc="Training Batches"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser(description="Train disease classifier")
    parser.add_argument('--data', type=str, default='data.csv', help='Path to CSV file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--hidden', nargs='+', type=int, default=[128, 128],
                        help='List of hidden layer sizes')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    args = parser.parse_args()

    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and prepare data
    print("Loading and preprocessing data...")
    X, y, label_encoder, scaler = load_data(args.data)
    train_loader, test_loader = make_dataloaders(
        X, y, batch_size=args.batch_size
    )

    # Build model
    model = Net(
        input_size=X.shape[1],
        hidden_sizes=args.hidden,
        output_size=len(label_encoder.classes_),
        dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5
    )

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        avg_loss = train(model, train_loader, criterion, optimizer, device)
        accuracy = evaluate(model, test_loader, device)
        print(f"Average Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.4f}")

    # Save artifacts
    print("Saving model, label encoder, and scaler...")
    torch.save(model.state_dict(), 'model.pth')
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("All done.")


if __name__ == "__main__":
    main()