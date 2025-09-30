import numpy as np

import torch
import torchvision
from torchvision import datasets
from torchvision import transforms

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()])

def get_data(data, indices=None, binarize=True):
  N = len(data)
  if indices is None:
    indices = range(0, N)
  #X = torch.stack([data[i][0] for i in indices], dim=1).squeeze(0) # (N,28,28)
  X = np.stack([data[i][0].numpy() for i in indices], axis=1).squeeze(0) # (N,28,28)
  if binarize: X = (X > 0.5)
  #y = torch.tensor([data[i][1] for i in indices])
  y = np.array([data[i][1] for i in indices])
  return X, y

data = datasets.EMNIST(
    root="~/data",
    split="balanced",
    download=True,
    transform=data_transform
)

X, y = get_data(data)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.90, random_state=0)
	
