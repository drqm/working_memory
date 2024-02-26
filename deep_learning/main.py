from conformer import *
from dataloading import *
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.optim as optim
import pickle


subject = '0011_U7X'
# recall_array_ls, manual_array_ls = get_subject_array_ls(subject)
# with open('data_arrays.pkl', 'wb') as f:
#     pickle.dump((recall_array_ls, manual_array_ls), f)
with open('data_arrays.pkl', 'rb') as f:
    recall_array_ls, manual_array_ls = pickle.load(f) # [(30, 306, 501), (29, 306, 501)] + [(30, 306, 501), (30, 306, 501)]

data_arrays = recall_array_ls + manual_array_ls

# labels

#Strategy 1
# recall_array_ls melody0&1 --> 0&1
# manual_array_ls melody0&1 --> 2&3
labels = [np.full((arr.shape[0],), i) for i, arr in enumerate(data_arrays, start=0)]
y = np.concatenate(labels, axis=0)

#Strategy 2
# recall_array_ls --> 0
# manual_array_ls --> 1
# recall_labels = [np.full((arr.shape[0],), 0) for arr in recall_array_ls]
# manual_labels = [np.full((arr.shape[0],), 1) for arr in manual_array_ls]
# labels = recall_labels + manual_labels
# y = np.concatenate(labels, axis=0)

#Strategy 3
# melody0 --> 0
# melody1 --> 1
# labels = []
# labels.append(np.full((recall_array_ls[0].shape[0],), 0))
# labels.append(np.full((recall_array_ls[1].shape[0],), 1))
# labels.append(np.full((manual_array_ls[0].shape[0],), 0))
# labels.append(np.full((manual_array_ls[1].shape[0],), 1))
# y = np.concatenate(labels, axis=0)




X = np.concatenate(data_arrays, axis=0)
X = X[:, None, :, 100:]
X = DElize(X, 10, 1)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

mean = X_train.mean()
std = X_train.std()
X_train_normalized = (X_train - mean) / std
X_test_normalized = (X_test - mean) / std

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16, shuffle=False)

model = Conformer()  # model choose comformer

# criterion and optimizer parameters
criterion_cls = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))

# def train&test functions
def train(model, device, train_loader, optimizer, criterion_cls, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        _, output = model(data)
        loss = criterion_cls(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def tes(model, device, test_loader, criterion_cls):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, output = model(data)
            test_loss += criterion_cls(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train and Test within epochs
for epoch in range(1, 20):
    train(model, device, train_loader, optimizer, criterion_cls, epoch)
    tes(model, device, test_loader, criterion_cls)