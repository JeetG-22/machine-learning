import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from task2 import CategoricalNaiveBayes
from emnist_project import X_train, X_test, y_train, y_test

# flatten the data
X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(float)
X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(float)

print(f"Training samples: {X_train_flat.shape[0]}")
print(f"Test samples: {X_test_flat.shape[0]}")


def plot_learning_curve_with_mle(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
    mle_train_scores=None,
    mle_test_scores=None
):
    """
    Plot learning curve with MLE comparison
    """
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(10, 6))

    axes.set_title(title, fontsize=14, fontweight='bold')
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training Set Size", fontsize=12)
    axes.set_ylabel("Score (Avg Log-Likelihood)", fontsize=12)

    # get the learning curve data
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        shuffle=False
    )

    # calculate mean and std
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    axes.grid(True, alpha=0.3)
    
    # plot MAP curves
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
        label="MAP Training", linewidth=2, markersize=8
    )
    axes.plot(
        train_sizes_abs, test_scores_mean, "s-", color="g", 
        label="MAP Validation", linewidth=2, markersize=8
    )
    
    # add MLE curves for comparison if provided
    if mle_train_scores is not None and mle_test_scores is not None:
        mle_train_mean = np.mean(mle_train_scores, axis=1)
        mle_test_mean = np.mean(mle_test_scores, axis=1)
        
        axes.plot(
            train_sizes_abs, mle_train_mean, "o--", color="orange", 
            label="MLE Training", linewidth=2, markersize=6, alpha=0.7
        )
        axes.plot(
            train_sizes_abs, mle_test_mean, "s--", color="blue", 
            label="MLE Validation", linewidth=2, markersize=6, alpha=0.7
        )
    
    axes.legend(loc="best", fontsize=9)

    return plt, train_sizes_abs, train_scores, test_scores


# we need 5 different training sizes from 10% to 100%
train_sizes = np.linspace(0.1, 1.0, 5)

# First, compute MLE scores once (we'll reuse for all plots)
print("\nComputing MLE baseline for comparison...")
model_mle = CategoricalNaiveBayes(alpha=1.0, beta=1.0, method='MLE')
_, mle_train_scores, mle_test_scores = learning_curve(
    model_mle,
    X_train_flat,
    y_train,
    cv=3,
    n_jobs=-1,
    train_sizes=train_sizes,
    shuffle=False
)
print("Done!")


# Task 3.1 - testing different alpha values with beta fixed at 1
print("\n" + "=" * 70)
print("TASK 3.1: Testing different alpha values")
print("=" * 70)

beta_fixed = 1.0
alphas_to_test = [1, 10, 50, 100, 200]

# make a grid of plots
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.ravel()

# test each alpha value
for idx, alpha in enumerate(alphas_to_test):
    print(f"Testing alpha = {alpha}...")
    
    model = CategoricalNaiveBayes(alpha=alpha, beta=beta_fixed, method='MAP')
    
    # plot with MLE comparison
    plot_learning_curve_with_mle(
        model,
        f"alpha={alpha}, beta={beta_fixed}",
        X_train_flat,
        y_train,
        axes=axes[idx],
        cv=3,
        n_jobs=-1,
        train_sizes=train_sizes,
        mle_train_scores=mle_train_scores,
        mle_test_scores=mle_test_scores
    )

# use last subplot for legend explanation
axes[5].text(0.5, 0.6, 'MLE vs MAP Comparison', 
             ha='center', va='center', fontsize=16, fontweight='bold',
             transform=axes[5].transAxes)
axes[5].text(0.5, 0.45, 'MLE (dashed lines) shown on all plots', 
             ha='center', va='center', fontsize=12,
             transform=axes[5].transAxes)
axes[5].text(0.5, 0.35, 'for direct comparison with MAP', 
             ha='center', va='center', fontsize=12,
             transform=axes[5].transAxes)
axes[5].axis('off')

plt.suptitle('Task 3.1: Effect of Class Prior alpha (beta=1 fixed)\nMLE shown for comparison on all plots', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('task3_1_alpha_curves.png', dpi=150, bbox_inches='tight')
print("\nSaved plot: task3_1_alpha_curves.png")
plt.show()


# Task 3.2 - testing different beta values with alpha fixed at 1
print("\n" + "=" * 70)
print("TASK 3.2: Testing different beta values")
print("=" * 70)

alpha_fixed = 1.0
betas_to_test = [1, 2, 10, 100]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

# test each beta
for idx, beta in enumerate(betas_to_test):
    print(f"Testing beta = {beta}...")
    
    model = CategoricalNaiveBayes(alpha=alpha_fixed, beta=beta, method='MAP')
    
    # plot with MLE comparison
    plot_learning_curve_with_mle(
        model,
        f"alpha={alpha_fixed}, beta={beta}",
        X_train_flat,
        y_train,
        axes=axes[idx],
        cv=3,
        n_jobs=-1,
        train_sizes=train_sizes,
        mle_train_scores=mle_train_scores,
        mle_test_scores=mle_test_scores
    )

plt.suptitle('Task 3.2: Effect of Pixel Prior beta (alpha=1 fixed)\nMLE shown for comparison on all plots', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('task3_2_beta_curves.png', dpi=150, bbox_inches='tight')
print("\nSaved plot: task3_2_beta_curves.png")
plt.show()


# make comparison plots to see everything together
print("\nMaking comparison plots...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# get MLE validation scores for comparison
_, _, mle_test_only = learning_curve(
    model_mle, X_train_flat, y_train,
    cv=3, n_jobs=-1, train_sizes=train_sizes, shuffle=False
)
mle_test_mean = np.mean(mle_test_only, axis=1)

# compare different alphas
print("Comparing alpha values...")
train_sizes_abs = None
for alpha in alphas_to_test:
    model = CategoricalNaiveBayes(alpha=alpha, beta=beta_fixed, method='MAP')
    
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X_train_flat, y_train,
        cv=3, n_jobs=-1, train_sizes=train_sizes, shuffle=False
    )
    
    test_scores_mean = np.mean(test_scores, axis=1)
    
    axes[0].plot(train_sizes_abs, test_scores_mean, 'o-', 
                 label=f'alpha={alpha}', linewidth=2, markersize=6)

# add MLE to comparison
axes[0].plot(train_sizes_abs, mle_test_mean, 's--', 
             label='MLE', linewidth=2, markersize=6, color='red', alpha=0.7)

axes[0].set_xlabel('Training Set Size', fontsize=12)
axes[0].set_ylabel('Validation Score', fontsize=12)
axes[0].set_title('Task 3.1: Validation Scores\n(varying alpha, beta=1 fixed)', 
                  fontsize=13, fontweight='bold')
axes[0].legend(loc='best', fontsize=10)
axes[0].grid(True, alpha=0.3)

# compare different betas
print("Comparing beta values...")
for beta in betas_to_test:
    model = CategoricalNaiveBayes(alpha=alpha_fixed, beta=beta, method='MAP')
    
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X_train_flat, y_train,
        cv=3, n_jobs=-1, train_sizes=train_sizes, shuffle=False
    )
    
    test_scores_mean = np.mean(test_scores, axis=1)
    
    axes[1].plot(train_sizes_abs, test_scores_mean, 's-', 
                 label=f'beta={beta}', linewidth=2, markersize=6)

# add MLE to comparison
axes[1].plot(train_sizes_abs, mle_test_mean, 'o--', 
             label='MLE', linewidth=2, markersize=6, color='red', alpha=0.7)

axes[1].set_xlabel('Training Set Size', fontsize=12)
axes[1].set_ylabel('Validation Score', fontsize=12)
axes[1].set_title('Task 3.2: Validation Scores\n(varying beta, alpha=1 fixed)', 
                  fontsize=13, fontweight='bold')
axes[1].legend(loc='best', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Task 3: Direct Comparison of Hyperparameters with MLE', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('task3_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved plot: task3_comparison.png")
plt.show()


print("\n" + "=" * 70)
print("Task 3 Complete!")
print("\nGenerated files:")
print("  - task3_1_alpha_curves.png (5 subplots, all with MLE comparison)")
print("  - task3_2_beta_curves.png (4 subplots, all with MLE comparison)")
print("  - task3_comparison.png (side-by-side with MLE)")
print("=" * 70)