import numpy as np
import matplotlib.pyplot as plt
from task2 import CategoricalNaiveBayes
from emnist_project import X_train, X_test, y_train, y_test

# flatten data
X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(float)
X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(float)

print(f"Training samples: {X_train_flat.shape[0]}")
print(f"Test samples: {X_test_flat.shape[0]}")


def create_imbalanced_dataset(X, y, total_size, alpha_class, random_state=42):
    """
    Creates an imbalanced dataset using Dirichlet distribution
    Smaller alpha_class = more imbalanced
    """
    np.random.seed(random_state)
    
    classes = np.unique(y)
    n_classes = len(classes)
    
    # sample proportions from Dirichlet
    class_proportions = np.random.dirichlet([alpha_class] * n_classes)
    
    # figure out how many samples per class
    target_counts = (class_proportions * total_size).astype(int)
    
    # fix rounding errors
    diff = total_size - np.sum(target_counts)
    if diff > 0:
        target_counts[0] += diff
    
    # sample from each class
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
    
    # shuffle
    shuffle_idx = np.random.permutation(len(X_imbalanced))
    X_imbalanced = X_imbalanced[shuffle_idx]
    y_imbalanced = y_imbalanced[shuffle_idx]
    
    return X_imbalanced, y_imbalanced, target_counts


def learning_curve(alpha, beta, alpha_class, method='MAP'):
    """
    Generate learning curve for given hyperparameters
    Using 3 training sizes for speed
    """
    train_sizes = np.array([0.1, 0.55, 1.0])  # start, middle, end
    
    train_scores = []
    val_scores = []
    actual_sizes = []
    
    for pct in train_sizes:
        n_samples = int(pct * len(X_train_flat))
        actual_sizes.append(n_samples)
        
        # create imbalanced dataset
        X_imb, y_imb, _ = create_imbalanced_dataset(
            X_train_flat, y_train, 
            total_size=n_samples,
            alpha_class=alpha_class
        )
        
        # train model
        model = CategoricalNaiveBayes(alpha=alpha, beta=beta, method=method)
        model.fit(X_imb, y_imb)
        
        # get scores
        train_scores.append(model.score(X_imb, y_imb))
        val_scores.append(model.score(X_test_flat, y_test))
    
    return np.array(train_scores), np.array(val_scores), np.array(actual_sizes)


def plot_class_distribution(class_counts, alpha_class):
    """Show how imbalanced the classes are"""
    plt.figure(figsize=(14, 4))
    plt.bar(range(len(class_counts)), class_counts)
    plt.xlabel('Class Index', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(f'Class Distribution (alpha_class={alpha_class})', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()


# First let's see what the different imbalance levels look like
print("\n" + "=" * 70)
print("Visualizing different imbalance levels...")
print("=" * 70)

for alpha_class in [0.1, 1, 100]:
    _, _, counts = create_imbalanced_dataset(
        X_train_flat, y_train, 
        total_size=11280,
        alpha_class=alpha_class
    )
    
    print(f"\nalpha_class = {alpha_class}:")
    print(f"  Min samples: {np.min(counts)}")
    print(f"  Max samples: {np.max(counts)}")
    print(f"  Std dev: {np.std(counts):.1f}")
    
    plot_class_distribution(counts, alpha_class)
    plt.savefig(f'task4_class_dist_alpha{alpha_class}.png', dpi=150, bbox_inches='tight')
    plt.close()


# TASK 4.1: Fix alpha = 1, vary beta
print("\n" + "=" * 70)
print("TASK 4.1: Testing beta with imbalanced data")
print("Note: Using 3 training sizes for speed optimization")
print("=" * 70)

alpha_fixed = 1.0
betas_to_test = [1, 1.2, 2, 10, 100]
alpha_class_values = [0.1, 0.2, 0.5, 1, 10, 100]  # all 6 levels

# Pre-compute MLE for all imbalance levels
print("\nPre-computing MLE baselines for comparison...")
mle_cache = {}
for alpha_class in alpha_class_values:
    train_scores_mle, val_scores_mle, sizes = learning_curve(
        alpha=1.0, beta=1.0, alpha_class=alpha_class, method='MLE'
    )
    mle_cache[alpha_class] = (train_scores_mle, val_scores_mle, sizes)
print("Done!")

for alpha_class in alpha_class_values:
    print(f"\n{'='*70}")
    print(f"Testing with alpha_class = {alpha_class}")
    if alpha_class < 1:
        print("  (highly imbalanced)")
    elif alpha_class < 5:
        print("  (somewhat imbalanced)")
    else:
        print("  (mostly balanced)")
    print(f"{'='*70}")
    
    # get MLE scores for this imbalance level
    train_scores_mle, val_scores_mle, sizes = mle_cache[alpha_class]
    
    # make grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    results = {}
    results['MLE'] = (train_scores_mle, val_scores_mle)
    
    # test each beta value
    for idx, beta in enumerate(betas_to_test):
        print(f"  Testing beta = {beta}...", end=' ')
        
        train_scores, val_scores, sizes = learning_curve(
            alpha=alpha_fixed,
            beta=beta,
            alpha_class=alpha_class,
            method='MAP'
        )
        
        results[f'beta={beta}'] = (train_scores, val_scores)
        
        # plot MAP curves
        axes[idx].plot(sizes, train_scores, 'o-', 
                      color='red', label='MAP Training', linewidth=2, markersize=6)
        axes[idx].plot(sizes, val_scores, 's-', 
                      color='green', label='MAP Validation', linewidth=2, markersize=6)
        
        # add MLE for comparison (dashed lines)
        axes[idx].plot(sizes, train_scores_mle, 'o--', 
                      color='orange', label='MLE Training', linewidth=2, markersize=4, alpha=0.7)
        axes[idx].plot(sizes, val_scores_mle, 's--', 
                      color='blue', label='MLE Validation', linewidth=2, markersize=4, alpha=0.7)
        
        axes[idx].set_xlabel('Training Size', fontsize=10)
        axes[idx].set_ylabel('Score', fontsize=10)
        axes[idx].set_title(f'beta={beta}', fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)
        
        print(f"done (val score: {val_scores[-1]:.4f})")
    
    print(f"  MLE: val score: {val_scores_mle[-1]:.4f}")
    
    # use last subplot for legend
    axes[5].text(0.5, 0.6, 'MLE vs MAP Comparison', 
                 ha='center', va='center', fontsize=14, fontweight='bold',
                 transform=axes[5].transAxes)
    axes[5].text(0.5, 0.45, 'MLE (dashed) vs MAP (solid)', 
                 ha='center', va='center', fontsize=11,
                 transform=axes[5].transAxes)
    axes[5].text(0.5, 0.35, 'shown on all subplots', 
                 ha='center', va='center', fontsize=11,
                 transform=axes[5].transAxes)
    axes[5].axis('off')
    
    # save the grid
    plt.suptitle(f'Task 4.1: beta with Imbalanced Data (alpha_class={alpha_class})\nMLE shown for comparison', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'task4_1_beta_alpha_class_{alpha_class}.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: task4_1_beta_alpha_class_{alpha_class}.png")
    plt.close()
    
    # make a comparison plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for label, (train_scores, val_scores) in results.items():
        linestyle = '--' if label == 'MLE' else '-'
        plt.plot(sizes, train_scores, 'o' + linestyle, label=label, linewidth=2, markersize=6)
    plt.xlabel('Training Size', fontsize=12)
    plt.ylabel('Training Score', fontsize=12)
    plt.title(f'Training (alpha_class={alpha_class})', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for label, (train_scores, val_scores) in results.items():
        linestyle = '--' if label == 'MLE' else '-'
        plt.plot(sizes, val_scores, 's' + linestyle, label=label, linewidth=2, markersize=6)
    plt.xlabel('Training Size', fontsize=12)
    plt.ylabel('Validation Score', fontsize=12)
    plt.title(f'Validation (alpha_class={alpha_class})', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Task 4.1: Comparison (alpha_class={alpha_class})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'task4_1_comparison_alpha_class_{alpha_class}.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: task4_1_comparison_alpha_class_{alpha_class}.png")
    plt.close()


# TASK 4.2: Fix beta = 1, vary alpha
print("\n" + "=" * 70)
print("TASK 4.2: Testing alpha with imbalanced data")
print("=" * 70)

beta_fixed = 1.0
alphas_to_test = [1, 10, 100, 1000]

for alpha_class in alpha_class_values:
    print(f"\n{'='*70}")
    print(f"Testing with alpha_class = {alpha_class}")
    print(f"{'='*70}")
    
    # get MLE scores for this imbalance level
    train_scores_mle, val_scores_mle, sizes = mle_cache[alpha_class]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    results = {}
    results['MLE'] = (train_scores_mle, val_scores_mle)
    
    # test each alpha
    for idx, alpha in enumerate(alphas_to_test):
        print(f"  Testing alpha = {alpha}...", end=' ')
        
        train_scores, val_scores, sizes = learning_curve(
            alpha=alpha,
            beta=beta_fixed,
            alpha_class=alpha_class,
            method='MAP'
        )
        
        results[f'alpha={alpha}'] = (train_scores, val_scores)
        
        # plot MAP curves
        axes[idx].plot(sizes, train_scores, 'o-', 
                      color='red', label='MAP Training', linewidth=2, markersize=6)
        axes[idx].plot(sizes, val_scores, 's-', 
                      color='green', label='MAP Validation', linewidth=2, markersize=6)
        
        # add MLE for comparison (dashed lines)
        axes[idx].plot(sizes, train_scores_mle, 'o--', 
                      color='orange', label='MLE Training', linewidth=2, markersize=4, alpha=0.7)
        axes[idx].plot(sizes, val_scores_mle, 's--', 
                      color='blue', label='MLE Validation', linewidth=2, markersize=4, alpha=0.7)
        
        axes[idx].set_xlabel('Training Size', fontsize=10)
        axes[idx].set_ylabel('Score', fontsize=10)
        axes[idx].set_title(f'alpha={alpha}', fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)
        
        print(f"done (val score: {val_scores[-1]:.4f})")
    
    print(f"  MLE: val score: {val_scores_mle[-1]:.4f}")
    
    # use remaining subplots for legend/notes
    axes[4].text(0.5, 0.6, 'MLE vs MAP Comparison', 
                 ha='center', va='center', fontsize=14, fontweight='bold',
                 transform=axes[4].transAxes)
    axes[4].text(0.5, 0.45, 'MLE (dashed) vs MAP (solid)', 
                 ha='center', va='center', fontsize=11,
                 transform=axes[4].transAxes)
    axes[4].text(0.5, 0.35, 'shown on all subplots', 
                 ha='center', va='center', fontsize=11,
                 transform=axes[4].transAxes)
    axes[4].axis('off')
    
    axes[5].axis('off')
    
    # save
    plt.suptitle(f'Task 4.2: alpha with Imbalanced Data (alpha_class={alpha_class})\nMLE shown for comparison', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'task4_2_alpha_alpha_class_{alpha_class}.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: task4_2_alpha_alpha_class_{alpha_class}.png")
    plt.close()
    
    # comparison plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for label, (train_scores, val_scores) in results.items():
        linestyle = '--' if label == 'MLE' else '-'
        plt.plot(sizes, train_scores, 'o' + linestyle, label=label, linewidth=2, markersize=6)
    plt.xlabel('Training Size', fontsize=12)
    plt.ylabel('Training Score', fontsize=12)
    plt.title(f'Training (alpha_class={alpha_class})', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for label, (train_scores, val_scores) in results.items():
        linestyle = '--' if label == 'MLE' else '-'
        plt.plot(sizes, val_scores, 's' + linestyle, label=label, linewidth=2, markersize=6)
    plt.xlabel('Training Size', fontsize=12)
    plt.ylabel('Validation Score', fontsize=12)
    plt.title(f'Validation (alpha_class={alpha_class})', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Task 4.2: Comparison (alpha_class={alpha_class})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'task4_2_comparison_alpha_class_{alpha_class}.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: task4_2_comparison_alpha_class_{alpha_class}.png")
    plt.close()


print("\n" + "=" * 70)
print("Task 4 Complete!")
print("\nSpeed optimizations used:")
print("  - I used 3 training sizes instead of 5 to make it faster")
print("  - I cached MLE results (avoids redundant computation)")
print("\nGenerated files:")
print("  - task4_class_dist_alpha*.png (3 files)")
print("  - task4_1_*.png (12 files - 6 grids + 6 comparisons)")
print("  - task4_2_*.png (12 files - 6 grids + 6 comparisons)")
print("  Total: 27 plots with MLE comparison on every subplot")
print("=" * 70)