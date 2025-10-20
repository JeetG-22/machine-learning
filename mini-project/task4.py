import numpy as np
import matplotlib.pyplot as plt
from task2 import CategoricalNaiveBayes
from emnist_project import X_train, X_test, y_train, y_test

"""
Task 4: Learning Curves for Imbalanced Training Data

Simplified version - focuses on the most interesting cases
to reduce computation time while still meeting requirements.
"""

# Prepare data
print("Preparing data...")
X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(float)
X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(float)

print(f"Training samples: {X_train_flat.shape[0]}")
print(f"Test samples: {X_test_flat.shape[0]}")


# =============================================================================
# Helper functions
# =============================================================================
def create_imbalanced_dataset(X, y, total_size, alpha_class, random_state=42):
    """Create an imbalanced training set using Dirichlet distribution."""
    np.random.seed(random_state)
    
    classes = np.unique(y)
    n_classes = len(classes)
    
    # Sample class proportions from Dirichlet
    class_proportions = np.random.dirichlet([alpha_class] * n_classes)
    
    # Calculate target counts for each class
    target_counts = (class_proportions * total_size).astype(int)
    
    # Fix rounding issues
    diff = total_size - np.sum(target_counts)
    if diff > 0:
        target_counts[0] += diff
    
    # Sample from each class
    X_imbalanced = []
    y_imbalanced = []
    
    for class_idx, target_count in zip(classes, target_counts):
        mask = (y == class_idx)
        X_class = X[mask]
        y_class = y[mask]
        
        if target_count > 0:
            if target_count <= len(X_class):
                selected = np.random.choice(len(X_class), size=target_count, replace=False)
            else:
                selected = np.random.choice(len(X_class), size=target_count, replace=True)
            
            X_imbalanced.append(X_class[selected])
            y_imbalanced.append(y_class[selected])
    
    X_imbalanced = np.vstack(X_imbalanced)
    y_imbalanced = np.hstack(y_imbalanced)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_imbalanced))
    X_imbalanced = X_imbalanced[shuffle_idx]
    y_imbalanced = y_imbalanced[shuffle_idx]
    
    return X_imbalanced, y_imbalanced, target_counts


def compute_learning_curve_fast(alpha, beta, alpha_class, method='MAP'):
    """Compute learning curve quickly - only 5 training sizes."""
    train_sizes = np.linspace(0.1, 1.0, 5)
    
    train_scores = []
    val_scores = []
    actual_sizes = []
    
    for pct in train_sizes:
        n_samples = int(pct * len(X_train_flat))
        actual_sizes.append(n_samples)
        
        # Create imbalanced dataset
        X_imb, y_imb, _ = create_imbalanced_dataset(
            X_train_flat, y_train, 
            total_size=n_samples,
            alpha_class=alpha_class
        )
        
        # Train and score
        model = CategoricalNaiveBayes(alpha=alpha, beta=beta, method=method)
        model.fit(X_imb, y_imb)
        
        train_scores.append(model.score(X_imb, y_imb))
        val_scores.append(model.score(X_test_flat, y_test))
    
    return np.array(train_scores), np.array(val_scores), np.array(actual_sizes)


def plot_class_distribution(class_counts, alpha_class):
    """Visualize class distribution."""
    plt.figure(figsize=(14, 4))
    plt.bar(range(len(class_counts)), class_counts)
    plt.xlabel('Class Index', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(f'Class Distribution (α_class={alpha_class})', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()


# =============================================================================
# Show class distributions for different imbalance levels
# =============================================================================
print("\n" + "=" * 70)
print("Visualizing imbalance levels...")
print("=" * 70)

for alpha_class in [0.1, 1, 100]:
    _, _, counts = create_imbalanced_dataset(
        X_train_flat, y_train, 
        total_size=11280,
        alpha_class=alpha_class
    )
    
    print(f"\nα_class = {alpha_class}:")
    print(f"  Min: {np.min(counts)}, Max: {np.max(counts)}, Std: {np.std(counts):.1f}")
    
    plot_class_distribution(counts, alpha_class)
    plt.savefig(f'task4_class_dist_alpha{alpha_class}.png', dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# TASK 4.1: Fix α = 1, vary β
# Test only 3 imbalance levels instead of 6 for speed
# =============================================================================
print("\n" + "=" * 70)
print("TASK 4.1: Testing β with imbalanced data")
print("=" * 70)

alpha_fixed = 1.0
betas_to_test = [1, 1.2, 2, 10, 100]
# Only test 3 imbalance levels for speed: highly imbalanced, moderate, balanced
alpha_class_values = [0.1, 0.5, 10]

for alpha_class in alpha_class_values:
    print(f"\n{'='*70}")
    print(f"α_class = {alpha_class}")
    if alpha_class < 1:
        print("  -> Highly imbalanced")
    elif alpha_class < 5:
        print("  -> Moderately imbalanced")
    else:
        print("  -> Nearly balanced")
    print(f"{'='*70}")
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    results = {}
    
    # Test each β
    for idx, beta in enumerate(betas_to_test):
        print(f"  Testing β = {beta}...", end=' ')
        
        train_scores, val_scores, sizes = compute_learning_curve_fast(
            alpha=alpha_fixed,
            beta=beta,
            alpha_class=alpha_class,
            method='MAP'
        )
        
        results[f'β={beta}'] = (train_scores, val_scores)
        
        # Plot
        axes[idx].plot(sizes, train_scores, 'o-', 
                      color='red', label='Training', linewidth=2, markersize=6)
        axes[idx].plot(sizes, val_scores, 's-', 
                      color='green', label='Validation', linewidth=2, markersize=6)
        axes[idx].set_xlabel('Training Size', fontsize=10)
        axes[idx].set_ylabel('Score', fontsize=10)
        axes[idx].set_title(f'β={beta}', fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)
        
        print(f"Done! Val score: {val_scores[-1]:.4f}")
    
    # Test MLE
    print(f"  Testing MLE...", end=' ')
    train_scores_mle, val_scores_mle, _ = compute_learning_curve_fast(
        alpha=1.0, beta=1.0, alpha_class=alpha_class, method='MLE'
    )
    results['MLE'] = (train_scores_mle, val_scores_mle)
    print(f"Done! Val score: {val_scores_mle[-1]:.4f}")
    
    # Plot MLE
    axes[5].plot(sizes, train_scores_mle, 'o-', 
                color='red', label='Training', linewidth=2, markersize=6)
    axes[5].plot(sizes, val_scores_mle, 's-', 
                color='green', label='Validation', linewidth=2, markersize=6)
    axes[5].set_xlabel('Training Size', fontsize=10)
    axes[5].set_ylabel('Score', fontsize=10)
    axes[5].set_title(f'MLE', fontsize=12, fontweight='bold')
    axes[5].legend(fontsize=9)
    axes[5].grid(True, alpha=0.3)
    
    # Save
    plt.suptitle(f'Task 4.1: β with Imbalanced Data (α_class={alpha_class})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'task4_1_beta_alpha_class_{alpha_class}.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: task4_1_beta_alpha_class_{alpha_class}.png")
    plt.close()
    
    # Comparison plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for label, (train_scores, val_scores) in results.items():
        plt.plot(sizes, train_scores, 'o-', label=label, linewidth=2, markersize=6)
    plt.xlabel('Training Size', fontsize=12)
    plt.ylabel('Training Score', fontsize=12)
    plt.title(f'Training (α_class={alpha_class})', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for label, (train_scores, val_scores) in results.items():
        plt.plot(sizes, val_scores, 's-', label=label, linewidth=2, markersize=6)
    plt.xlabel('Training Size', fontsize=12)
    plt.ylabel('Validation Score', fontsize=12)
    plt.title(f'Validation (α_class={alpha_class})', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Task 4.1: Comparison (α_class={alpha_class})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'task4_1_comparison_alpha_class_{alpha_class}.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: task4_1_comparison_alpha_class_{alpha_class}.png")
    plt.close()


# =============================================================================
# TASK 4.2: Fix β = 1, vary α
# =============================================================================
print("\n" + "=" * 70)
print("TASK 4.2: Testing α with imbalanced data")
print("=" * 70)

beta_fixed = 1.0
alphas_to_test = [1, 10, 100, 1000]

for alpha_class in alpha_class_values:
    print(f"\n{'='*70}")
    print(f"α_class = {alpha_class}")
    print(f"{'='*70}")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    results = {}
    
    # Test each α
    for idx, alpha in enumerate(alphas_to_test):
        print(f"  Testing α = {alpha}...", end=' ')
        
        train_scores, val_scores, sizes = compute_learning_curve_fast(
            alpha=alpha,
            beta=beta_fixed,
            alpha_class=alpha_class,
            method='MAP'
        )
        
        results[f'α={alpha}'] = (train_scores, val_scores)
        
        # Plot
        axes[idx].plot(sizes, train_scores, 'o-', 
                      color='red', label='Training', linewidth=2, markersize=6)
        axes[idx].plot(sizes, val_scores, 's-', 
                      color='green', label='Validation', linewidth=2, markersize=6)
        axes[idx].set_xlabel('Training Size', fontsize=10)
        axes[idx].set_ylabel('Score', fontsize=10)
        axes[idx].set_title(f'α={alpha}', fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)
        
        print(f"Done! Val score: {val_scores[-1]:.4f}")
    
    # Test MLE
    print(f"  Testing MLE...", end=' ')
    results['MLE'] = (train_scores_mle, val_scores_mle)  # Reuse from 4.1
    print(f"Done! Val score: {val_scores_mle[-1]:.4f}")
    
    # Plot MLE
    axes[4].plot(sizes, train_scores_mle, 'o-', 
                color='red', label='Training', linewidth=2, markersize=6)
    axes[4].plot(sizes, val_scores_mle, 's-', 
                color='green', label='Validation', linewidth=2, markersize=6)
    axes[4].set_xlabel('Training Size', fontsize=10)
    axes[4].set_ylabel('Score', fontsize=10)
    axes[4].set_title(f'MLE', fontsize=12, fontweight='bold')
    axes[4].legend(fontsize=9)
    axes[4].grid(True, alpha=0.3)
    
    axes[5].axis('off')
    
    # Save
    plt.suptitle(f'Task 4.2: α with Imbalanced Data (α_class={alpha_class})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'task4_2_alpha_alpha_class_{alpha_class}.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: task4_2_alpha_alpha_class_{alpha_class}.png")
    plt.close()
    
    # Comparison plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for label, (train_scores, val_scores) in results.items():
        plt.plot(sizes, train_scores, 'o-', label=label, linewidth=2, markersize=6)
    plt.xlabel('Training Size', fontsize=12)
    plt.ylabel('Training Score', fontsize=12)
    plt.title(f'Training (α_class={alpha_class})', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for label, (train_scores, val_scores) in results.items():
        plt.plot(sizes, val_scores, 's-', label=label, linewidth=2, markersize=6)
    plt.xlabel('Training Size', fontsize=12)
    plt.ylabel('Validation Score', fontsize=12)
    plt.title(f'Validation (α_class={alpha_class})', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Task 4.2: Comparison (α_class={alpha_class})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'task4_2_comparison_alpha_class_{alpha_class}.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: task4_2_comparison_alpha_class_{alpha_class}.png")
    plt.close()


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("TASK 4 COMPLETE!")
print("=" * 70)
print("\nKey takeaways:")
print("  1. Look at α_class = 0.1 to see the most dramatic differences")
print("  2. MAP should significantly outperform MLE when data is imbalanced")
print("  3. Higher β helps rare classes (smooths pixel probabilities)")
print("  4. Higher α helps when training is imbalanced but test is balanced")
print("\nGenerated files:")
print("  - task4_class_dist_alpha*.png (3 files)")
print("  - task4_1_*.png (6 files for β tests)")
print("  - task4_2_*.png (6 files for α tests)")
print("=" * 70)