import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms

print("=" * 60)
print("UNDERSTANDING THE EMNIST DATASET")
print("=" * 60)

# STEP 1: Load without any transformations first
print("\n1. LOADING RAW DATA (no transformations)...")
raw_data = datasets.EMNIST(
    root="./data",
    split="balanced",
    download=True,
    transform=transforms.ToTensor()  # Just convert to tensor, no rotation
)

print(f"✓ Total samples in dataset: {len(raw_data)}")
print(f"✓ Number of classes: {len(raw_data.classes)}")
print(f"✓ Class names: {raw_data.classes[:20]}... (showing first 20)")

# STEP 2: Look at a single sample
print("\n2. EXAMINING A SINGLE SAMPLE...")
img, label = raw_data[0]  # Get the first sample

print(f"✓ Image type: {type(img)}")
print(f"✓ Image shape: {img.shape}")  # Should be [1, 28, 28]
print(f"   - {img.shape[0]} = color channels (1 for grayscale)")
print(f"   - {img.shape[1]} = height (28 pixels)")
print(f"   - {img.shape[2]} = width (28 pixels)")
print(f"✓ Label: {label} (which means '{raw_data.classes[label]}')")
print(f"✓ Pixel value range: [{img.min().item():.3f}, {img.max().item():.3f}]")

# STEP 3: Show what raw EMNIST looks like (WRONG orientation)
print("\n3. VISUALIZING RAW vs CORRECTED IMAGES...")
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Top Row: RAW EMNIST (wrong orientation)\nBottom Row: CORRECTED (rotated & flipped)")

for i in range(5):
    # Get a sample
    img_raw, label = raw_data[i * 1000]  # Sample every 1000th image
    
    # Show raw (incorrect orientation)
    axes[0, i].imshow(img_raw.squeeze(), cmap='gray')
    axes[0, i].set_title(f"Label: {raw_data.classes[label]}")
    axes[0, i].axis('off')
    
    # Show corrected
    img_corrected = torchvision.transforms.functional.rotate(img_raw, -90)
    img_corrected = torchvision.transforms.functional.hflip(img_corrected)
    axes[1, i].imshow(img_corrected.squeeze(), cmap='gray')
    axes[1, i].set_title(f"Corrected: {raw_data.classes[label]}")
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('emnist_raw_vs_corrected.png', dpi=150, bbox_inches='tight')
print("✓ Saved comparison to 'emnist_raw_vs_corrected.png'")

# STEP 4: Load with proper transformations
print("\n4. LOADING DATA WITH TRANSFORMATIONS...")
transform = transforms.Compose([
    lambda img: torchvision.transforms.functional.rotate(img, -90),
    transforms.RandomHorizontalFlip(p=1),
    transforms.ToTensor()
])

corrected_data = datasets.EMNIST(
    root="./data",
    split="balanced",
    download=False,  # Already downloaded
    transform=transform
)

# STEP 5: Show class distribution
print("\n5. CLASS DISTRIBUTION IN DATASET...")
class_counts = {}
for i in range(len(corrected_data)):
    _, label = corrected_data[i]
    class_counts[label] = class_counts.get(label, 0) + 1

print(f"✓ Samples per class (first 10 classes):")
for class_id in range(min(20, len(class_counts))):
    class_name = corrected_data.classes[class_id]
    count = class_counts[class_id]
    print(f"   Class {class_id:2d} ('{class_name}'): {count:5d} samples")

# STEP 6: Show how to access samples
print("\n6. HOW TO ACCESS SAMPLES...")
print("You can access samples like a Python list:")
print(f"  training_data[0]     → (image, label) for first sample")
print(f"  training_data[100]   → (image, label) for 101st sample")
print(f"  len(training_data)   → {len(corrected_data)} total samples")

# Example: Get multiple samples
print("\nExample - Getting samples for class '5':")
samples_of_five = []
for idx in range(len(corrected_data)):
    img, label = corrected_data[idx]
    if corrected_data.classes[label] == '5' and len(samples_of_five) < 3:
        samples_of_five.append((idx, img, label))
    if len(samples_of_five) >= 3:
        break

for idx, img, label in samples_of_five:
    print(f"  Index {idx}: label={label}, shape={img.shape}")

# STEP 7: Visualize how data is structured
print("\n7. DATA STRUCTURE SUMMARY...")
print("┌─────────────────────────────────────────┐")
print("│ training_data = datasets.EMNIST(...)   │")
print("└─────────────────────────────────────────┘")
print("           │")
print("           │ Acts like a list")
print("           ▼")
print("  training_data[0] → (image, label)")
print("           │             │      │")
print("           │             │      └─→ Integer: 0-46")
print("           │             │")
print("           │             └─→ Tensor: [1, 28, 28]")
print("           │")
print("  training_data[1] → (image, label)")
print("           │")
print("         ...")
print("           │")
print(f"  training_data[{len(corrected_data)-1}] → (image, label)")

print("\n" + "=" * 60)
print("Now you understand the EMNIST dataset!")
print("=" * 60)
plt.show()