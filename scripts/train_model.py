import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from models.cnn_model import CNN

class BreathingDataset(Dataset):
    def __init__(self, data_list, le):
        self.X = []
        self.y = []
        for item in data_list:
            flow = item['Flow']
            thorac = item['Thorac']
            spo2_raw = item['SpO2']
            
            x_spo2 = np.linspace(0, 30, len(spo2_raw))
            x_target = np.linspace(0, 30, len(flow))
            spo2_interp = np.interp(x_target, x_spo2, spo2_raw)
            
            signal = np.vstack([flow, thorac, spo2_interp])
            self.X.append(signal)
            self.y.append(item['Label'])
            
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(le.transform(self.y), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Trainer:
    def __init__(self, model, device, optimizer, loss_fn):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, train_loader, epochs=20):
        self.model.train()
        for epoch in range(epochs):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

    def evaluate(self, test_loader):
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        return all_targets, all_preds

def plot_confusion_matrix(cm, classes, test_ap):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f"{test_ap}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"Visualizations/confusion_matrix_{test_ap}.png")
    plt.close()

def main():
    with open('Dataset/dataset.pkl', 'rb') as f:
        full_dataset = pickle.load(f)

    all_labels = [item['Label'] for item in full_dataset]
    le = LabelEncoder()
    le.fit(all_labels)
    num_classes = len(le.classes_)
    class_names = le.classes_

    participants = list(set([item['AP'] for item in full_dataset]))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    for test_ap in participants:
        print(f"participant: {test_ap}")
        
        train_data = [item for item in full_dataset if item['AP'] != test_ap]
        test_data = [item for item in full_dataset if item['AP'] == test_ap]

        train_dataset = BreathingDataset(train_data, le)
        test_dataset = BreathingDataset(test_data, le)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = CNN(num_classes).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        trainer = Trainer(model, device, optimizer, loss_fn)
        trainer.train(train_loader, epochs=10)
        
        all_targets, all_preds = trainer.evaluate(test_loader)

        acc = accuracy_score(all_targets, all_preds)
        prec = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        rec = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        cm = confusion_matrix(all_targets, all_preds)

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print("-" * 30)

        plot_confusion_matrix(cm, class_names, test_ap)

if __name__ == "__main__":
    main()