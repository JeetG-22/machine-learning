# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

import superimport

import numpy as np
import matplotlib.pyplot as plt


import torch
import torchvision
from torchvision import datasets
from torchvision import transforms

# We need to rotate the images
# https://github.com/pytorch/vision/issues/2630

transform=transforms.Compose([lambda img: torchvision.transforms.functional.rotate(img, -90),
                                transforms.RandomHorizontalFlip(p=1),
                                transforms.ToTensor()])

training_data = datasets.EMNIST(
    root="./data",
    split="balanced",
    download=True,
    transform=transform
)

# Organize samples by class
print("Organizing samples by class...")
samples_by_class = {}
for idx in range(len(training_data)):
    _, label = training_data[idx]
    if label not in samples_by_class:
        samples_by_class[label] = []
    samples_by_class[label].append(idx)

# Task 1: 5 rows x C columns (where C=47)
rows = 5
cols = len(training_data.classes)  # 47 classes

figure = plt.figure(figsize=(cols * 0.7, rows * 0.7))

# For each position in the grid
for i in range(1, cols * rows + 1):
    # Calculate which class (column) and which sample (row)
    col = (i - 1) % cols  # class ID
    row = (i - 1) // cols  # sample number for this class
    
    # Get a random sample from this specific class
    class_samples = samples_by_class[col]
    sample_idx = np.random.choice(class_samples)
    
    img, label = training_data[sample_idx]
    
    figure.add_subplot(rows, cols, i)
    label_name = training_data.classes[label]
    
    # Only show title on first row
    if row == 0:
        plt.title(label_name, fontsize=8)
    
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img.squeeze(), cmap=plt.cm.binary)

plt.tight_layout()
plt.savefig('task1_viz.png', dpi=150, bbox_inches='tight')
print(f"Saved visualization: {rows} rows x {cols} columns")
plt.show()