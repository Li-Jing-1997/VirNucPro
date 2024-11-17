# 对每个种类单独训练效果还可以
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import os
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_auc_score

random.seed(42)
class FileBatchDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.data = []
        self.labels = []
        self._load_all_data()

    def _load_all_data(self):
        for file_path in self.file_list:
            data_dict = torch.load(file_path)
            data = data_dict['data'][:, :]
            label = data_dict['labels'][0]
            self.data.append(data)
            self.labels.append(label)

    def __len__(self):
        return sum(d.size(0) for d in self.data)

    def __getitem__(self, idx):
        cumulative_size = 0
        for data, label in zip(self.data, self.labels):
            if cumulative_size + data.size(0) > idx:
                index_in_file = idx - cumulative_size
                sample = data[index_in_file]
                return sample, torch.tensor(label)
            cumulative_size += data.size(0)
        raise IndexError("Index out of range")

target_subdirectory = 'viral.1.1_merged'
folder_list = sorted([os.path.join('./data/data_merge/', f) for f in os.listdir('./data/data_merge/') if os.path.isdir(os.path.join('./data/data_merge/', f))])

viral_list = []
other_list = []

for folder in folder_list:
    standardized_path = os.path.normpath(folder)
    path_parts = standardized_path.split(os.sep)
    if target_subdirectory in path_parts:
        viral_list.extend(
            sorted([os.path.join(folder, f) for f in os.listdir(folder)
                    if f.endswith('.pt')])
        )
    else:
        other_list.extend(
            sorted([os.path.join(folder, f) for f in os.listdir(folder)
                    if f.endswith('.pt')])
        )
file_list = viral_list + other_list
dataset = FileBatchDataset(file_list)
total_size = len(dataset)
indices = list(range(total_size))
split = int(total_size * 0.1)

random.shuffle(indices)

test_indices = indices[:split]
train_indices = indices[split:]

train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=12)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class):
        super(MLPClassifier, self).__init__()

        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.output_layer = nn.Linear(hidden_dim, num_class)

        self.dropout = nn.Dropout(0.5)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.hidden_layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

       
input_dim = 3328
hidden_dim = 512
num_class = 2
mlp_model = MLPClassifier(input_dim, hidden_dim, num_class)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp_model = mlp_model.to(device)

def write_log(filename, message):
    with open(filename, 'a') as f:
        f.write(message + '\n')

criterion = nn.CrossEntropyLoss()
mlp_optimizer = optim.SGD(mlp_model.parameters(), lr=0.0002, momentum=0.9)
scheduler = StepLR(mlp_optimizer, step_size=10, gamma=0.85)
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop_counter = 0
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_model_wts = model.state_dict()
        elif val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.best_model_wts = model.state_dict()
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter >= self.patience:
                print("Early stopping triggered")
                model.load_state_dict(self.best_model_wts)
                return True
        return False

def train_model(model, optimizer, train_loader, test_loader, num_epochs=200, log_file='MLP_log.txt', patience=5):
    early_stopping = EarlyStopping(patience=patience)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_data, batch_labels in tqdm(train_loader, total=len(train_loader)):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        log_message = f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}'
        print(log_message)
        write_log(log_file, log_message)

        val_loss, accuracy, precision, recall, f1, auc = test_model(model, test_loader, log_file)

        if early_stopping(val_loss, model):
            torch.save(model, 'model.pth')
            break

        if scheduler:
            scheduler.step()

        

def test_model(model, test_loader, log_file='MLP_log.txt'):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []
    total_loss = 0.0
    correct_predictions = 0

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_data)
            loss = F.cross_entropy(outputs, batch_labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(batch_labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(F.softmax(outputs, dim=1)[:, 1].cpu().numpy())

            correct_predictions += (predicted == batch_labels).sum().item()

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')
    avg_loss = total_loss / len(test_loader)
    auc = roc_auc_score(all_labels, all_probs)

    log_message = (f'Test Loss: {avg_loss:.4f}\n'
                   f'Test Accuracy: {accuracy:.4f}\n'
                   f'Test Precision: {precision:.4f}\n'
                   f'Test Recall: {recall:.4f}\n'
                   f'Test F1 Score: {f1:.4f}\n'
                   f'Test AUROC: {auc:.4f}\n'
                   f'Correct predictions: {correct_predictions}/{len(all_labels)}')

    print(log_message)
    write_log(log_file, log_message)
    
    return avg_loss, accuracy, precision, recall, f1, auc

train_model(mlp_model, mlp_optimizer, train_loader, test_loader)
