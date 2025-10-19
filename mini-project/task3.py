import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from task2 import CategoricalNaiveBayes
from emnist_project import X_train, X_test, y_train, y_test

"""
Task 3: Learning Curves for Balanced Training Data

Following the sklearn example from:
https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

From the project description:
- Use 5 training sizes: 10%, 32.5%, 55%, 77.5%, 100% of 11,280 samples
- Compare MLE vs MAP with different hyperparameters
- Plot training score and validation score for each

From Lecture 3: Learning curves help us understand overfitting and 
the effect of training data size on model performance.
"""

# Prepare data
print("Preparing data...")
X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(float)
X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(float)

print(f"Training samples: {X_train_flat.shape[0]}")
print(f"Test samples: {X_test_flat.shape[0]}")


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
    scoring=None
):
    """
    Generate learning curve plots.
    
    Adapted from sklearn documentation:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    
    From Lecture 2 & 3: Learning curves show how model performance 
    changes with training set size. Useful for diagnosing:
    - Overfitting (large gap between train and validation)
    - Underfitting (both scores are poor)
    - Benefit of more data (curves converging upward)
    
    Parameters
    ----------
    estimator : object
        The ML model to evaluate (our CategoricalNaiveBayes)
    title : str
        Title for the chart
    X : array-like
        Training data
    y : array-like
        Target labels
    axes : matplotlib axes, optional
        Axes to draw plot on
    ylim : tuple, optional
        Y-axis limits
    cv : int or cross-validation generator
        Number of folds for cross-validation
    n_jobs : int, optional
        Number of jobs to run in parallel
    train_sizes : array-like
        Relative or absolute numbers of training examples
    scoring : str, optional
        Scoring method (we'll use default which calls .score())
    """
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(10, 6))

    axes.set_title(title, fontsize=14, fontweight='bold')
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training Set Size", fontsize=12)
    axes.set_ylabel("Score (Avg Log-Likelihood)", fontsize=12)

    # Use sklearn's learning_curve function
    # This automatically handles train/test splitting and scoring
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        scoring=scoring,
        shuffle=False  # Don't shuffle to maintain consistency
    )

    # Compute mean and std for train and test scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve with shaded error regions
    axes.grid(True, alpha=0.3)
    axes.fill_between(
        train_sizes_abs,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes.fill_between(
        train_sizes_abs,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes.plot(
        train_sizes_abs, train_scores_mean, "o-", color="r", 
        label="Training score", linewidth=2, markersize=8
    )
    axes.plot(
        train_sizes_abs, test_scores_mean, "s-", color="g", 
        label="Validation score", linewidth=2, markersize=8
    )
    axes.legend(loc="best", fontsize=11)

    return plt


# =============================================================================
# Task 3.1: Fix β = 1, vary α = {1, 10, 50, 100, 200}
# =============================================================================
print("\n" + "=" * 70)
print("TASK 3.1: Effect of Class Prior (α) on Learning")
print("=" * 70)
print("From Lecture 3: α controls Dirichlet prior on class distribution")
print("Higher α = stronger belief that classes should be balanced")
print("=" * 70)

# Fix beta, vary alpha
beta_fixed = 1.0
alphas_to_test = [1, 10, 50, 100, 200]

# Training sizes: 10%, 32.5%, 55%, 77.5%, 100%
train_sizes = np.linspace(0.1, 1.0, 5)

# Create figure with subplots for each alpha value
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()  # Flatten to 1D array for easy indexing

for idx, alpha in enumerate(alphas_to_test):
    print(f"\nTesting α = {alpha}, β = {beta_fixed} (MAP)...")
    
    # Create model with these hyperparameters
    model = CategoricalNaiveBayes(alpha=alpha, beta=beta_fixed, method='MAP')
    
    # Generate learning curve
    # cv=3 means 3-fold cross-validation for more robust estimates
    plot_learning_curve(
        model,
        f"Learning Curve (α={alpha}, β={beta_fixed})",
        X_train_flat,
        y_train,
        axes=axes[idx],
        cv=3,  # 3-fold cross-validation
        n_jobs=-1,  # Use all CPU cores
        train_sizes=train_sizes
    )

# Also add MLE for comparison
print(f"\nTesting MLE (for comparison)...")
model_mle = CategoricalNaiveBayes(alpha=1.0, beta=1.0, method='MLE')
plot_learning_curve(
    model_mle,
    f"Learning Curve (MLE)",
    X_train_flat,
    y_train,
    axes=axes[5],
    cv=3,
    n_jobs=-1,
    train_sizes=train_sizes
)

plt.suptitle('Task 3.1: Effect of Class Prior α (β=1 fixed)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('task3_1_alpha_curves.png', dpi=150, bbox_inches='tight')
print("\nSaved plot: task3_1_alpha_curves.png")
plt.show()


# =============================================================================
# Task 3.2: Fix α = 1, vary β = {1, 2, 10, 100}
# =============================================================================
print("\n" + "=" * 70)
print("TASK 3.2: Effect of Pixel Prior (β) on Learning")
print("=" * 70)
print("From Lecture 3: β controls Beta prior on pixel probabilities")
print("Higher β = stronger smoothing toward θ = 0.5")
print("=" * 70)

# Fix alpha, vary beta
alpha_fixed = 1.0
betas_to_test = [1, 2, 10, 100]

# Create figure with subplots for each beta value
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, beta in enumerate(betas_to_test):
    print(f"\nTesting α = {alpha_fixed}, β = {beta} (MAP)...")
    
    # Create model with these hyperparameters
    model = CategoricalNaiveBayes(alpha=alpha_fixed, beta=beta, method='MAP')
    
    # Generate learning curve
    plot_learning_curve(
        model,
        f"Learning Curve (α={alpha_fixed}, β={beta})",
        X_train_flat,
        y_train,
        axes=axes[idx],
        cv=3,
        n_jobs=-1,
        train_sizes=train_sizes
    )

# Add MLE for comparison
axes[4].text(0.5, 0.5, 'MLE shown in Task 3.1', 
             ha='center', va='center', fontsize=14,
             transform=axes[4].transAxes)
axes[4].axis('off')

# Leave last subplot empty
axes[5].axis('off')

plt.suptitle('Task 3.2: Effect of Pixel Prior β (α=1 fixed)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('task3_2_beta_curves.png', dpi=150, bbox_inches='tight')
print("\nSaved plot: task3_2_beta_curves.png")
plt.show()


# =============================================================================
# Create a direct comparison plot (all on same axes)
# =============================================================================
print("\n" + "=" * 70)
print("Creating comparison plots...")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Comparison for Task 3.1 (varying alpha)
print("\nComparing different α values...")
for alpha in alphas_to_test:
    model = CategoricalNaiveBayes(alpha=alpha, beta=beta_fixed, method='MAP')
    
    # Get learning curve data
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X_train_flat, y_train,
        cv=3, n_jobs=-1, train_sizes=train_sizes, shuffle=False
    )
    
    # Plot only test scores for cleaner comparison
    test_scores_mean = np.mean(test_scores, axis=1)
    axes[0].plot(train_sizes_abs, test_scores_mean, 'o-', 
                 label=f'α={alpha}', linewidth=2, markersize=6)

axes[0].set_xlabel('Training Set Size', fontsize=12)
axes[0].set_ylabel('Validation Score', fontsize=12)
axes[0].set_title('Task 3.1: Validation Scores\n(varying α, β=1 fixed)', 
                  fontsize=13, fontweight='bold')
axes[0].legend(loc='best', fontsize=10)
axes[0].grid(True, alpha=0.3)

# Comparison for Task 3.2 (varying beta)
print("Comparing different β values...")
for beta in betas_to_test:
    model = CategoricalNaiveBayes(alpha=alpha_fixed, beta=beta, method='MAP')
    
    # Get learning curve data
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X_train_flat, y_train,
        cv=3, n_jobs=-1, train_sizes=train_sizes, shuffle=False
    )
    
    # Plot only test scores for cleaner comparison
    test_scores_mean = np.mean(test_scores, axis=1)
    axes[1].plot(train_sizes_abs, test_scores_mean, 's-', 
                 label=f'β={beta}', linewidth=2, markersize=6)

axes[1].set_xlabel('Training Set Size', fontsize=12)
axes[1].set_ylabel('Validation Score', fontsize=12)
axes[1].set_title('Task 3.2: Validation Scores\n(varying β, α=1 fixed)', 
                  fontsize=13, fontweight='bold')
axes[1].legend(loc='best', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Task 3: Direct Comparison of Hyperparameters', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('task3_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved plot: task3_comparison.png")
plt.show()


# =============================================================================
# Analysis Summary
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS GUIDE")
print("=" * 70)

print("\nWhat to look for in the learning curves:")
print("-" * 70)
print("1. Training vs Validation Gap:")
print("   From Lecture 2 (Overfitting):")
print("   - Large gap → overfitting (model memorizes training data)")
print("   - Small gap → good generalization")
print("   - MAP should reduce this gap compared to MLE")
print()
print("2. Effect of Training Data Size:")
print("   From Lecture 3 (Asymptotic behavior):")
print("   - More data → better scores")
print("   - MAP → MLE as N → ∞ (prior becomes less important)")
print("   - Curves should converge")
print()
print("3. Effect of α (Dirichlet prior on classes):")
print("   From Lecture 3:")
print("   - Low α (≈1): Similar to MLE")
print("   - High α (≥100): Strong regularization toward uniform classes")
print("   - With balanced data, all α should perform similarly")
print()
print("4. Effect of β (Beta prior on pixels):")
print("   From Lecture 3:")
print("   - β = 1: MLE-like (no smoothing)")
print("   - β = 2: Slight smoothing (often optimal)")
print("   - β ≥ 10: Heavy smoothing (may underfit)")
print("   - β prevents zero probabilities!")
print()
print("5. Cross-Validation:")
print("   Following sklearn example:")
print("   - Shaded regions show std deviation across CV folds")
print("   - Gives more reliable performance estimates")
print("   - Wider bands → more variability in performance")
print()
print("Expected Results:")
print("   - With balanced data, α variations shouldn't differ much")
print("   - β = 2 or β = 10 should outperform β = 1 (MLE)")
print("   - High β (100) might hurt performance (over-regularization)")
print("=" * 70)

print("\n" + "=" * 70)
print("Task 3 Complete!")
print("=" * 70)
print("\nGenerated files:")
print("  - task3_1_alpha_curves.png (individual plots for each α)")
print("  - task3_2_beta_curves.png (individual plots for each β)")
print("  - task3_comparison.png (direct comparison on same axes)")
print("=" * 70)