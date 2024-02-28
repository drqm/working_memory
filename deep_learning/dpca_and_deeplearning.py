from conformer import *
from dataloading import *
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.optim as optim
import pickle
from dpca_calculation import dpca_fit

subject = '0011_U7X'
recall_array_ls, manual_array_ls = get_subject_array_ls(subject)
with open('data_arrays.pkl', 'wb') as f:
    pickle.dump((recall_array_ls, manual_array_ls), f)
with open('data_arrays.pkl', 'rb') as f:
    recall_array_ls, manual_array_ls = pickle.load(f) # [(30, 306, 501), (29, 306, 501)] + [(30, 306, 501), (30, 306, 501)]

data_arrays = recall_array_ls + manual_array_ls

# reshape data for dpca
min_experiment_count = min([i.shape[0] for i in data_arrays])
new_matrix_shape = [min_experiment_count, 306, 2, 2, 501] # [29, 306, 2, 2, 501]
new_matrix = np.zeros(new_matrix_shape)

for i in range(2):
    for j in range(2):
        new_matrix[:, :, i, j, :] = data_arrays[i*2+j][:min_experiment_count]

# split train test data
experiment_indices = np.arange(min_experiment_count)
train_indices, test_indices = train_test_split(experiment_indices, test_size=0.3)

train_matrix = new_matrix[train_indices] #(20, 306, 2, 2, 501)
test_matrix = new_matrix[test_indices] #(9, 306, 2, 2, 501)

n_components = 5
Z, dpca = dpca_fit(train_matrix, 'bst', n_components)

def fit_and_sort_array(train_matrix):
    new_array = np.zeros([len(train_matrix),7*n_components,2,2,501]) #[20,7*5,2,2,501]
    for i in range(len(train_matrix)):
        a = dpca.transform(train_matrix[i])

        ncount = 0
        for j in a:
            new_array[i][ncount*5:ncount*5+5] = a[j]
            ncount = ncount + 1

    # [20*2*2,7*5,501] --> back to n_trials, n_channels, n_times again for training
    new_array = new_array.reshape(len(train_matrix)*2*2, 7*n_components, 501)
    print(new_array.shape)
    labels = np.tile(np.arange(4), len(train_matrix))

    return new_array, labels

X_train, y_train = fit_and_sort_array(train_matrix)
X_test, y_test = fit_and_sort_array(test_matrix)

X_train = X_train[:, None, :, 100:]
X_test = X_test[:, None, :, 100:]

#Standardize
mean = X_train.mean()
std = X_train.std()
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=8, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=8, shuffle=False)

with open('data_loader.pkl', 'wb') as f:
    pickle.dump((train_loader, test_loader), f)
with open('data_loader.pkl', 'rb') as f:
    train_loader, test_loader = pickle.load(f)

model = Conformer()  # model choose comformer

# criterion and optimizer parameters
criterion_cls = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

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
            # print('output,',output)
            # print('target,',target)
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