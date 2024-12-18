import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
from tqdm import tqdm
import scipy
from nltk.corpus import reuters
import nltk
nltk.download('wordnet')
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from yellowbrick.text import TSNEVisualizer
import warnings
warnings.filterwarnings("ignore")
import torch
from torch import nn, optim
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics



def prepare_text_data(texts_train, texts_test, labels_train, labels_test):
    tfidf = TfidfVectorizer(max_features=10000)
    X_train = tfidf.fit_transform(texts_train).toarray()
    X_test = tfidf.transform(texts_test).toarray()

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(labels_train)
    y_test = mlb.transform(labels_test)

    return X_train, X_test, y_train, y_test, tfidf, mlb


class ANN(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, n_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class MultiLabelTrainer:
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.1, verbose=True
        )

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for batch_x, batch_y in tqdm(train_loader, desc="Training"):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, data_loader, threshold=0.5):
        self.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0

        with torch.no_grad():
            for batch_x, batch_y in tqdm(data_loader, desc="Evaluating"):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()

                predictions = (outputs >= threshold).float()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # Calculate metrics
        f1 = f1_score(all_targets, all_predictions, average='macro')
        precision = precision_score(all_targets, all_predictions, average='macro')
        recall = recall_score(all_targets, all_predictions, average='macro')
        accuracy = accuracy_score(all_targets, all_predictions)

        return {
            'loss': total_loss / len(data_loader),
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }


def prepare_data(X_train, y_train, X_test, y_test, batch_size=32):
    if scipy.sparse.issparse(X_train):
        X_train = X_train.toarray()
    if scipy.sparse.issparse(X_test):
        X_test = X_test.toarray()

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader


def train_model(X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_features = X_train.shape[1]
    n_classes = y_train.shape[1]

    train_loader, test_loader = prepare_data(
        X_train, y_train, X_test, y_test, batch_size
    )

    model = ANN(n_features, n_classes).to(device)
    trainer = MultiLabelTrainer(model, device)

    best_train_loss = float('inf')
    best_model = None

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss = trainer.train_epoch(train_loader)

        # Evaluate on training set
        train_metrics = trainer.evaluate(train_loader)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Train F1: {train_metrics['f1']:.4f}")
        print(f"Train Precision: {train_metrics['precision']:.4f}")
        print(f"Train Recall: {train_metrics['recall']:.4f}")

        # Learning rate scheduling
        trainer.scheduler.step(train_loss)

        # Save best model based on training loss
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_model = model.state_dict().copy()

    # Load best model and evaluate on test set
    model.load_state_dict(best_model)
    test_metrics = trainer.evaluate(test_loader)

    print("\nTest Set Results:")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")

    return model, test_metrics


# Usage:
"""
# X_train: TF-IDF features for training
# X_test: TF-IDF features for testing
# y_train: Multi-label encoded labels for training
# y_test: Multi-label encoded labels for testing

# Train the model
model, metrics = train_model(
    X_train, y_train,
    X_test, y_test,
    epochs=50,
    batch_size=32
)

# Save the model
torch.save({
    'model_state_dict': model.state_dict(),
}, 'model_checkpoint.pth')
"""