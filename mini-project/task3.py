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
    
    This function:
    1. Trains the model on different amounts of data (10% to 100%)
    2. Uses cross-validation to get reliable estimates
    3. Plots training and validation scores with error bands
    
    From Lecture 2 & 3: Learning curves show:
    - Overfitting: large gap between train and validation
    - Underfitting: both scores are poor
    - Benefit of more data: curves converging upward
    
    Parameters
    ----------
    estimator : object
        The ML model (our CategoricalNaiveBayes)
    title : str
        Title for the chart
    X : array-like
        Training data
    y : array-like
        Target labels
    axes : matplotlib axes
        Where to draw the plot
    ylim : tuple
        Y-axis limits (optional)
    cv : int
        Number of cross-validation folds (e.g., cv=3 for 3-fold CV)
    n_jobs : int
        Number of parallel jobs (-1 means use all CPU cores)
    train_sizes : array-like
        Training set sizes to test (can be fractions or absolute numbers)
    scoring : str
        Scoring method (None uses the model's .score() method)
    """
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(10, 6))

    axes.set_title(title, fontsize=14, fontweight='bold')
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training Set Size", fontsize=12)
    axes.set_ylabel("Score (Avg Log-Likelihood)", fontsize=12)

    # Use sklearn's learning_curve function
    # This handles: train/test splitting, different training sizes, cross-validation
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,              # Cross-validation folds
        n_jobs=n_jobs,      # Parallel processing
        train_sizes=train_sizes,  # Which training sizes to test
        scoring=scoring,
        shuffle=False       # Keep data order consistent
    )

    # Compute statistics across CV folds
    # train_scores has shape (n_sizes, n_folds), we average across folds
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot with shaded error regions
    # From sklearn example: shaded areas show variability across CV folds
    axes.grid(True, alpha=0.3)
    
    # Training score: red with light red shading for ±1 std
    axes.fill_between(
        train_sizes_abs,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    
    # Validation score: green with light green shading
    axes.fill_between(
        train_sizes_abs,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    
    # Plot the mean curves on top
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

# Settings
beta_fixed = 1.0
alphas_to_test = [1, 10, 50, 100, 200]

# Training sizes: 10%, 32.5%, 55%, 77.5%, 100%
# From project description: use 5 evenly spaced points
train_sizes = np.linspace(0.1, 1.0, 5)

# Create figure with subplots (2 rows, 3 columns = 6 subplots)
# We need 5 for different α values + 1 for MLE
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()  # Flatten 2D array to 1D for easy indexing

# Test each α value
for idx, alpha in enumerate(alphas_to_test):
    print(f"\nTesting α = {alpha}, β = {beta_fixed} (MAP)...")
    
    # Create model with these hyperparameters
    model = CategoricalNaiveBayes(alpha=alpha, beta=beta_fixed, method='MAP')
    
    # Generate learning curve using sklearn's function
    # cv=3: split data 3 ways for cross-validation (more reliable estimates)
    # n_jobs=-1: use all CPU cores for speed
    plot_learning_curve(
        model,
        f"Learning Curve (α={alpha}, β={beta_fixed})",
        X_train_flat,
        y_train,
        axes=axes[idx],
        cv=3,
        n_jobs=-1,
        train_sizes=train_sizes
    )

# Also add MLE for comparison (last subplot)
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

# Add overall title and save
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

# Settings
alpha_fixed = 1.0
betas_to_test = [1, 2, 10, 100]

# Create figure with subplots (2 rows, 3 columns)
# We need 4 for different β values + 2 empty
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

# Test each β value
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

# Add note about MLE in empty subplot
axes[4].text(0.5, 0.5, 'MLE shown in Task 3.1', 
             ha='center', va='center', fontsize=14,
             transform=axes[4].transAxes)
axes[4].axis('off')

# Turn off last subplot (we only need 4)
axes[5].axis('off')

# Save the plot
plt.suptitle('Task 3.2: Effect of Pixel Prior β (α=1 fixed)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('task3_2_beta_curves.png', dpi=150, bbox_inches='tight')
print("\nSaved plot: task3_2_beta_curves.png")
plt.show()


# =============================================================================
# Create a direct comparison plot (all curves on same axes for easier comparison)
# =============================================================================
print("\n" + "=" * 70)
print("Creating comparison plots...")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ----------------------
# Comparison for Task 3.1 (varying alpha)
# ----------------------
print("\nComparing different α values...")
for alpha in alphas_to_test:
    model = CategoricalNaiveBayes(alpha=alpha, beta=beta_fixed, method='MAP')
    
    # Get learning curve data
    # This returns: (train_sizes, train_scores, test_scores)
    # Each score array has shape (n_sizes, n_folds)
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X_train_flat, y_train,
        cv=3, n_jobs=-1, train_sizes=train_sizes, shuffle=False
    )
    
    # Average across CV folds
    test_scores_mean = np.mean(test_scores, axis=1)
    
    # Plot only validation scores for cleaner comparison
    axes[0].plot(train_sizes_abs, test_scores_mean, 'o-', 
                 label=f'α={alpha}', linewidth=2, markersize=6)

# Format the plot
axes[0].set_xlabel('Training Set Size', fontsize=12)
axes[0].set_ylabel('Validation Score', fontsize=12)
axes[0].set_title('Task 3.1: Validation Scores\n(varying α, β=1 fixed)', 
                  fontsize=13, fontweight='bold')
axes[0].legend(loc='best', fontsize=10)
axes[0].grid(True, alpha=0.3)

# ----------------------
# Comparison for Task 3.2 (varying beta)
# ----------------------
print("Comparing different β values...")
for beta in betas_to_test:
    model = CategoricalNaiveBayes(alpha=alpha_fixed, beta=beta, method='MAP')
    
    # Get learning curve data
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X_train_flat, y_train,
        cv=3, n_jobs=-1, train_sizes=train_sizes, shuffle=False
    )
    
    # Average across CV folds
    test_scores_mean = np.mean(test_scores, axis=1)
    
    # Plot only validation scores
    axes[1].plot(train_sizes_abs, test_scores_mean, 's-', 
                 label=f'β={beta}', linewidth=2, markersize=6)

# Format the plot
axes[1].set_xlabel('Training Set Size', fontsize=12)
axes[1].set_ylabel('Validation Score', fontsize=12)
axes[1].set_title('Task 3.2: Validation Scores\n(varying β, α=1 fixed)', 
                  fontsize=13, fontweight='bold')
axes[1].legend(loc='best', fontsize=10)
axes[1].grid(True, alpha=0.3)

# Save comparison plot
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

print("\n1. Training vs Validation Gap:")
print("   From Lecture 2 (Overfitting):")
print("   - Large gap → overfitting (model memorizes training data)")
print("   - Small gap → good generalization")
print("   - MAP should reduce this gap compared to MLE")

print("\n2. Effect of Training Data Size:")
print("   From Lecture 3 (Asymptotic behavior):")
print("   - More data → better scores")
print("   - MAP → MLE as N → ∞ (prior becomes less important)")
print("   - Curves should converge as we add more data")

print("\n3. Effect of α (Dirichlet prior on classes):")
print("   From Lecture 3:")
print("   - Low α (≈1): Similar to MLE")
print("   - High α (≥100): Strong regularization toward uniform classes")
print("   - With balanced data, all α should perform similarly")
print("   - Would matter more with imbalanced classes (Task 4!)")

print("\n4. Effect of β (Beta prior on pixels):")
print("   From Lecture 3:")
print("   - β = 1: MLE-like (no smoothing)")
print("   - β = 2: Slight smoothing (often optimal)")
print("   - β ≥ 10: Heavy smoothing (may underfit)")
print("   - Key benefit: β prevents zero probabilities!")

print("\n5. Cross-Validation:")
print("   Following sklearn example:")
print("   - Shaded regions show std deviation across CV folds")
print("   - Gives more reliable performance estimates")
print("   - Wider bands → more variability in performance")

print("\n6. Expected Results:")
print("   - With balanced data, α variations shouldn't differ much")
print("   - β = 2 or β = 10 should outperform β = 1 (MLE)")
print("   - High β (100) might hurt performance (over-regularization)")

print("\n" + "=" * 70)
print("Task 3 Complete!")
print("=" * 70)
print("\nGenerated files:")
print("  - task3_1_alpha_curves.png (grid of 6 subplots for each α + MLE)")
print("  - task3_2_beta_curves.png (grid of subplots for each β)")
print("  - task3_comparison.png (direct comparison on same axes)")
print("=" * 70)