import sys
sys.path.append('src')
sys.path.append('deep_learning')

import importlib
from conformer import *
import conformer
importlib.reload(conformer)

from dataloading import *
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
import torch.optim as optim
import pickle
from dpca_calculation import dpca_fit

subject = '0011_U7X'
if len(sys.argv) > 1:
    subject = sys.argv[1]

period = 'imagine'
if len(sys.argv) > 2:
    period = sys.argv[2]

all_n_classes = [2,4]
if len(sys.argv) > 3:
    all_n_classes = [int(sys.argv[3])]

flips = [False,True]
if len(sys.argv) > 4:
    flip = [bool(sys.argv[4])]
     
periods = {'baseline': [-2,0],'listen': [0,2], 'imagine': [2,4]}

recall_array_ls, manual_array_ls, times, Fs = get_subject_array_ls(subject,crop=periods[period]) # also get time and Fs
data_arrays = recall_array_ls + manual_array_ls

# reshape data for dpca
nch, nt = recall_array_ls[0].shape[1:]
min_experiment_count = min([i.shape[0] for i in data_arrays])
new_matrix_shape = [min_experiment_count, nch, 2, 2, nt] # [29, 306, 2, 2, 501]


# split train test data
experiment_indices = np.arange(min_experiment_count)
n_folds= 3
n_iter = 30
accuracy = np.zeros((n_folds,n_iter)) * np.nan
kf = KFold(n_folds,shuffle=True)
kf.get_n_splits(experiment_indices)

for n_classes in all_n_classes:
	for flip in flips:
		new_matrix = np.zeros(new_matrix_shape)
		for i in range(2):
			for j in range(2):
				new_matrix[:, :, i, j, :] = data_arrays[i*2+j][:min_experiment_count]

		if flip:
			new_matrix[:, :, 1] = np.flip(new_matrix[:, :, 1],axis=3)

		# m_shuffle = False
		# b_shuffle = False
		# if m_shuffle:
		#     mix = np.random.choice(range(min_experiment_count),min_experiment_count//2)
		#     new_matrix[mix] = np.flip(new_matrix[mix],axis=3) 

		# if b_shuffle:
		#     bix = np.random.choice(range(min_experiment_count),min_experiment_count//2)
		#     new_matrix[bix] = np.flip(new_matrix[bix],axis=2) 

		for i, (train_indices, test_indices) in enumerate(kf.split(experiment_indices)):
			print(f"\nsubject {subject}, period {period}, flip {flip}, classes {n_classes}, Fold {i}:\n")
			
			train_matrix = new_matrix[train_indices] #(20, 306, 2, 2, 501)
			test_matrix = new_matrix[test_indices] #(9, 306, 2, 2, 501)

			n_components = 5
			Z, dpca = dpca_fit(train_matrix, 'bst', n_components)

			def fit_and_sort_array(cmatrix):
				new_array = np.zeros([len(cmatrix),7*n_components,2,2,nt]) #[20,7*5,2,2,501]
				for i in range(len(cmatrix)):
					a = dpca.transform(cmatrix[i])

					ncount = 0
					for j in a:
						new_array[i][ncount*5:ncount*5+5] = a[j]
						ncount = ncount + 1

				# [20*2*2,7*5,501] --> back to n_trials, n_channels, n_times again for training
				### Needs transposing first!
				new_array = new_array.transpose([0,2,3,1,4]).reshape(len(cmatrix)*2*2, 7*n_components, nt)
				#new_array = new_array.reshape(len(cmatrix)*2*2, 7*n_components, nt)
				print(new_array.shape)
				labels = np.tile(np.arange(4), len(cmatrix))

				return new_array, labels
			
			X_train, y_train = fit_and_sort_array(train_matrix)
			X_test, y_test = fit_and_sort_array(test_matrix)

			if n_classes == 2:
				y_train = y_train % 2
				y_test = y_test % 2

			#np.random.shuffle(y_train)
			#np.random.shuffle(y_test)
			tmin,tmax = periods[period]
			tix = (times>=tmin) & (times <= tmax)
			X_train = X_train[:, None, :, tix]
			X_test = X_test[:, None, :, tix]

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

			# with open('data_loader.pkl', 'wb') as f:
			#     pickle.dump((train_loader, test_loader), f)
			# with open('data_loader.pkl', 'rb') as f:
			#     train_loader, test_loader = pickle.load(f)

			model = conformer.Conformer(n_classes=n_classes)  # model choose comformer

			# criterion and optimizer parameters
			criterion_cls = nn.CrossEntropyLoss()
			optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

			# def train&test functions
			def train(model, device, ctrain_loader, optimizer, criterion_cls, epoch):
				model.train()
				for batch_idx, (train_data, train_target) in enumerate(ctrain_loader):
					train_data, train_target = train_data.to(device), train_target.to(device)
					optimizer.zero_grad()
					_, train_output = model(train_data)
					loss = criterion_cls(train_output, train_target)
					loss.backward()
					optimizer.step()
					if batch_idx % 10 == 0:
						print(f'Train Epoch: {epoch} [{batch_idx * len(train_data)}/{len(ctrain_loader.dataset)} ({100. * batch_idx / len(ctrain_loader):.0f}%)]\tLoss: {loss.item():.6f}')
				return model
		
			def test(model, device, ctest_loader, criterion_cls):
				model.eval()
				test_loss = 0
				correct = 0
				with torch.no_grad():
					for test_data, test_target in ctest_loader:
						test_data, test_target = test_data.to(device), test_target.to(device)
						_, test_output = model(test_data)
						# print('output,',output)
						# print('target,',target)
						test_loss += criterion_cls(test_output, test_target).item()  # sum up batch loss
						pred = test_output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
						correct += pred.eq(test_target.view_as(pred)).sum().item()

				test_loss /= len(ctest_loader.dataset)
				acc = correct / len(ctest_loader.dataset)
				print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(ctest_loader.dataset)} ({100. * correct / len(ctest_loader.dataset):.0f}%)\n')
				return acc
			
			# Device
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			model.to(device)

			# Train and Test within epochs
			for epoch in range(1, n_iter):
				train(model, device, train_loader, optimizer, criterion_cls, epoch)
				accuracy[i,epoch] = test(model, device, test_loader, criterion_cls)

		save_accuracy(accuracy, subject, f'_{period}_{flip}_{n_classes}')
	
