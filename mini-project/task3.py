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


# this function is from the sklearn documentation example
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
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(10, 6))

    axes.set_title(title, fontsize=14, fontweight='bold')
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training Set Size", fontsize=12)
    axes.set_ylabel("Score (Avg Log-Likelihood)", fontsize=12)

    # this gets the learning curve data using sklearn
    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        scoring=scoring,
        shuffle=False
    )

    # calculate mean and std across the folds
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # plot the results
    axes.grid(True, alpha=0.3)
    
    # shaded regions show the std deviation
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
    
    # plot the actual lines
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


# Task 3.1 - testing different alpha values with beta fixed at 1
print("\n")
print("TASK 3.1: Testing different alpha values")
print("=" * 70)

beta_fixed = 1.0
alphas_to_test = [1, 10, 50, 100, 200]

# we need 5 different training sizes from 10% to 100%
train_sizes = np.linspace(0.1, 1.0, 5)

# make a grid of plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

# test each alpha value
for idx, alpha in enumerate(alphas_to_test):
    print(f"\nTesting alpha = {alpha}...")
    
    model = CategoricalNaiveBayes(alpha=alpha, beta=beta_fixed, method='MAP')
    
    plot_learning_curve(
        model,
        f"Learning Curve (alpha={alpha}, beta={beta_fixed})",
        X_train_flat,
        y_train,
        axes=axes[idx],
        cv=3,
        n_jobs=-1,
        train_sizes=train_sizes
    )

# also add MLE for comparison
print(f"\nTesting MLE...")
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

plt.suptitle('Task 3.1: Effect of Class Prior alpha (beta=1 fixed)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('task3_1_alpha_curves.png', dpi=150, bbox_inches='tight')
print("\nSaved plot: task3_1_alpha_curves.png")
plt.show()


# Task 3.2 - testing different beta values with alpha fixed at 1
print("\n")
print("TASK 3.2: Testing different beta values")
print("=" * 70)

alpha_fixed = 1.0
betas_to_test = [1, 2, 10, 100]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

# test each beta
for idx, beta in enumerate(betas_to_test):
    print(f"\nTesting beta = {beta}...")
    
    model = CategoricalNaiveBayes(alpha=alpha_fixed, beta=beta, method='MAP')
    
    plot_learning_curve(
        model,
        f"Learning Curve (alpha={alpha_fixed}, beta={beta})",
        X_train_flat,
        y_train,
        axes=axes[idx],
        cv=3,
        n_jobs=-1,
        train_sizes=train_sizes
    )

# just put a note in the empty subplot
axes[4].text(0.5, 0.5, 'MLE shown in Task 3.1', 
             ha='center', va='center', fontsize=14,
             transform=axes[4].transAxes)
axes[4].axis('off')

axes[5].axis('off')

plt.suptitle('Task 3.2: Effect of Pixel Prior beta (alpha=1 fixed)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('task3_2_beta_curves.png', dpi=150, bbox_inches='tight')
print("\nSaved plot: task3_2_beta_curves.png")
plt.show()


# make comparison plots to see everything together
print("\nMaking comparison plots...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# compare different alphas
print("Comparing alpha values...")
for alpha in alphas_to_test:
    model = CategoricalNaiveBayes(alpha=alpha, beta=beta_fixed, method='MAP')
    
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X_train_flat, y_train,
        cv=3, n_jobs=-1, train_sizes=train_sizes, shuffle=False
    )
    
    test_scores_mean = np.mean(test_scores, axis=1)
    
    axes[0].plot(train_sizes_abs, test_scores_mean, 'o-', 
                 label=f'alpha={alpha}', linewidth=2, markersize=6)

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

axes[1].set_xlabel('Training Set Size', fontsize=12)
axes[1].set_ylabel('Validation Score', fontsize=12)
axes[1].set_title('Task 3.2: Validation Scores\n(varying beta, alpha=1 fixed)', 
                  fontsize=13, fontweight='bold')
axes[1].legend(loc='best', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Task 3: Direct Comparison of Hyperparameters', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('task3_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved plot: task3_comparison.png")
plt.show()


print("\n" + "=" * 70)
print("Task 3 Complete!")
print("\nGenerated files:")
print("  - task3_1_alpha_curves.png")
print("  - task3_2_beta_curves.png")
print("  - task3_comparison.png")
print("=" * 70)