import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class CategoricalNaiveBayes(BaseEstimator, ClassifierMixin):
    """
    Categorical Naive Bayes classifier for binary features (EMNIST).
    
    Based on Lecture 2 (Naive Bayes) and Lecture 3 (Bayesian Learning)
    
    Parameters:
    -----------
    alpha : float, default=1.0
        Dirichlet prior for class distribution (Lecture 3: Dir(α,...,α))
        Higher α = more smoothing toward uniform class distribution
    beta : float, default=1.0
        Beta prior for pixel probabilities (Lecture 3: Beta(β,β))
        Higher β = more smoothing toward 0.5 for each pixel
    method : str, default='MAP'
        'MLE' = Maximum Likelihood (Lecture 2)
        'MAP' = Maximum A Posteriori (Lecture 3)
    """
    
    def __init__(self, alpha=1.0, beta=1.0, method='MAP'):
        self.alpha = alpha
        self.beta = beta
        self.method = method
        
    def fit(self, X, y):
        """
        Learn model parameters from training data.
        
        From Lecture 2: We model p(x, y) = p(x|y) * p(y)
        - p(y) = class prior (which character is it?)
        - p(x|y) = likelihood (what pixels light up given the character?)
        """
        # Flatten images if needed: (n_samples, 28, 28) -> (n_samples, 784)
        if X.ndim > 2:
            n_samples = X.shape[0]
            X = X.reshape(n_samples, -1)
        
        # Store dimensions
        self.n_features_ = X.shape[1]  # 784 pixels
        self.classes_ = np.unique(y)    # [0, 1, 2, ..., 46] for EMNIST balanced
        self.n_classes_ = len(self.classes_)  # 47 classes
        
        # Step 1: Compute class probabilities π_c = P(y = c)
        # Lecture 3: Using either MLE or MAP with Dirichlet prior
        if self.method == 'MLE':
            self.class_log_prior_ = self._compute_class_prior_mle(y)
        else:  # MAP
            self.class_log_prior_ = self._compute_class_prior_map(y)
        
        # Step 2: Compute pixel probabilities θ_{d,c} = P(pixel_d = 1 | y = c)
        # Lecture 2: Naive Bayes assumes pixels are independent given class
        if self.method == 'MLE':
            self.feature_log_prob_ = self._compute_pixel_prob_mle(X, y)
        else:  # MAP
            self.feature_log_prob_ = self._compute_pixel_prob_map(X, y)
        
        return self
    
    def _compute_class_prior_mle(self, y):
        """
        MLE for class probabilities.
        
        From Lecture 2: π_c^{MLE} = N_c / N
        where N_c = number of samples in class c
        """
        # Count how many samples belong to each class
        class_counts = np.zeros(self.n_classes_)
        for idx, c in enumerate(self.classes_):
            class_counts[idx] = np.sum(y == c)  # Count samples where y == c
        
        # MLE: just divide by total number of samples
        total_samples = len(y)
        class_prob = class_counts / total_samples
        
        # Return log probabilities (numerical stability - Lecture 1)
        return np.log(class_prob)
    
    def _compute_class_prior_map(self, y):
        """
        MAP for class probabilities with Dirichlet prior.
        
        From Lecture 3: 
        Prior: π ~ Dir(α, α, ..., α)
        Posterior: π | D ~ Dir(α + N_0, α + N_1, ..., α + N_{C-1})
        MAP (mode): π_c = (N_c + α - 1) / (N + C*α - C)
        """
        # Count samples per class
        class_counts = np.zeros(self.n_classes_)
        for idx, c in enumerate(self.classes_):
            class_counts[idx] = np.sum(y == c)
        
        # MAP estimate with Dirichlet prior
        # Think of α as "pseudo-counts" we add to each class
        numerator = class_counts + self.alpha - 1
        denominator = len(y) + self.n_classes_ * self.alpha - self.n_classes_
        
        class_prob = numerator / denominator
        
        return np.log(class_prob)
    
    def _compute_pixel_prob_mle(self, X, y):
        """
        MLE for pixel probabilities.
        
        From Lecture 2:
        For Bernoulli distribution: θ^{MLE} = N_1 / (N_0 + N_1)
        
        For each pixel d and class c:
        θ_{d,c}^{MLE} = (# times pixel d is 1 in class c) / (# samples in class c)
        """
        # Initialize: one probability for each (class, pixel) pair
        pixel_prob = np.zeros((self.n_classes_, self.n_features_))
        
        # For each class, compute pixel probabilities
        for idx, c in enumerate(self.classes_):
            # Get all training images from this class
            X_c = X[y == c]  # Shape: (N_c, 784)
            
            # MLE: θ_{d,c} = (sum of pixel d values) / (number of samples)
            # Since pixels are binary, mean = fraction of 1s
            pixel_prob[idx, :] = np.mean(X_c, axis=0)
            
            # Avoid log(0) errors by clipping to small values
            # This prevents -inf when taking log later
            pixel_prob[idx, :] = np.clip(pixel_prob[idx, :], 1e-10, 1 - 1e-10)
        
        return np.log(pixel_prob)
    
    def _compute_pixel_prob_map(self, X, y):
        """
        MAP for pixel probabilities with Beta prior.
        
        From Lecture 3:
        Prior: θ_{d,c} ~ Beta(β, β)
        Posterior: θ_{d,c} | D ~ Beta(β + N_{dc1}, β + N_{dc0})
        MAP (mode): θ_{d,c} = (N_{dc1} + β - 1) / (N_c + 2β - 2)
        
        where N_{dc1} = count of 1s for pixel d in class c
              N_{dc0} = count of 0s for pixel d in class c
              N_c = N_{dc1} + N_{dc0}
        """
        # Initialize pixel probabilities
        pixel_prob = np.zeros((self.n_classes_, self.n_features_))
        
        for idx, c in enumerate(self.classes_):
            # Get all training images from this class
            X_c = X[y == c]  # Shape: (N_c, 784)
            n_samples_c = X_c.shape[0]  # N_c
            
            # Count how many times each pixel is 1
            count_ones = np.sum(X_c, axis=0)  # N_{dc1} for each pixel d
            
            # MAP with Beta prior
            # Think of β as "pseudo-observations" we add
            # β = 2 means we've seen 1 "on" and 1 "off" before seeing data
            numerator = count_ones + self.beta - 1
            denominator = n_samples_c + 2 * self.beta - 2
            
            pixel_prob[idx, :] = numerator / denominator
            
            # Still clip for safety (though MAP rarely hits 0 or 1)
            pixel_prob[idx, :] = np.clip(pixel_prob[idx, :], 1e-10, 1 - 1e-10)
        
        return np.log(pixel_prob)
    
    def predict(self, X):
        """
        Predict class labels for test samples.
        
        From Lecture 2: Use Bayes' Rule
        ŷ = argmax_c p(y = c | x)
          = argmax_c p(x | y = c) * p(y = c)  [drop p(x) since same for all c]
          = argmax_c log p(x, y = c)  [work in log space]
        """
        # Flatten images if needed
        if X.ndim > 2:
            n_samples = X.shape[0]
            X = X.reshape(n_samples, -1)
        
        # Compute log p(x, y = c) for all classes
        log_probs = self._joint_log_likelihood(X)
        
        # Pick class with highest probability
        class_indices = np.argmax(log_probs, axis=1)
        return self.classes_[class_indices]
    
    def _joint_log_likelihood(self, X):
        """
        Compute log p(x, y = c) for each class c.
        
        From Lecture 2 (Naive Bayes):
        p(x, y = c) = p(y = c) * p(x | y = c)
                    = π_c * ∏_{d=1}^{784} θ_{d,c}^{x_d} * (1-θ_{d,c})^{1-x_d}
        
        In log space:
        log p(x, y = c) = log π_c + Σ_{d=1}^{784} [x_d*log(θ_{d,c}) + (1-x_d)*log(1-θ_{d,c})]
        """
        n_samples = X.shape[0]
        log_probs = np.zeros((n_samples, self.n_classes_))
        
        # For each class
        for idx in range(self.n_classes_):
            # Start with class prior: log p(y = c)
            log_probs[:, idx] = self.class_log_prior_[idx]
            
            # Add likelihood: log p(x | y = c)
            # For each pixel: if pixel is 1, add log(θ), if 0, add log(1-θ)
            
            # Get log probabilities for this class
            log_theta = self.feature_log_prob_[idx, :]  # log(θ_{d,c}) for all pixels
            log_one_minus_theta = np.log(1 - np.exp(log_theta))  # log(1 - θ_{d,c})
            
            # For each sample, sum contributions from all pixels
            # Naive Bayes: pixels are independent given class
            for n in range(n_samples):
                # When pixel is 1: add log(θ)
                # When pixel is 0: add log(1-θ)
                contribution = X[n] * log_theta + (1 - X[n]) * log_one_minus_theta
                log_probs[n, idx] += np.sum(contribution)
        
        return log_probs
    
    def score(self, X, y):
        """
        Compute average log-likelihood of the data.
        
        From project description:
        Score = (1/N) * Σ_n log p(x_n, y_n | θ)
        
        Higher score = better fit to data
        Used to compare MLE vs MAP and detect overfitting
        """
        # Flatten images if needed
        if X.ndim > 2:
            n_samples = X.shape[0]
            X = X.reshape(n_samples, -1)
        
        # Get log probabilities for all classes
        log_probs = self._joint_log_likelihood(X)  # Shape: (n_samples, n_classes)
        
        # For each sample, extract the log probability of its TRUE class
        log_likelihood_per_sample = []
        for n in range(len(y)):
            true_label = y[n]
            # Find which class index corresponds to this label
            class_idx = np.where(self.classes_ == true_label)[0][0]
            # Get log p(x_n, y_n = true_label)
            log_likelihood_per_sample.append(log_probs[n, class_idx])
        
        # Return average log-likelihood
        return np.mean(log_likelihood_per_sample)


# Test the implementation
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Categorical Naive Bayes Implementation")
    print("=" * 60)
    
    # Load data (from your emnist_project.py)
    from emnist_project import X_train, X_test, y_train, y_test
    
    # Flatten images: (n_samples, 28, 28) -> (n_samples, 784)
    X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(float)
    X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(float)
    
    print(f"\nDataset Info:")
    print(f"  Training samples: {X_train_flat.shape[0]}")
    print(f"  Test samples: {X_test_flat.shape[0]}")
    print(f"  Features (pixels): {X_train_flat.shape[1]}")
    print(f"  Classes: {len(np.unique(y_train))}")
    
    # Train MLE model
    print("\n" + "-" * 60)
    print("Training MLE Model (no regularization)...")
    print("-" * 60)
    model_mle = CategoricalNaiveBayes(alpha=1.0, beta=1.0, method='MLE')
    model_mle.fit(X_train_flat, y_train)
    
    # Train MAP model
    print("\nTraining MAP Model (with Bayesian priors)...")
    print("-" * 60)
    model_map = CategoricalNaiveBayes(alpha=1.0, beta=1.0, method='MAP')
    model_map.fit(X_train_flat, y_train)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred_mle = model_mle.predict(X_test_flat)
    y_pred_map = model_map.predict(X_test_flat)
    
    # Evaluate accuracy
    from sklearn.metrics import accuracy_score
    acc_mle = accuracy_score(y_test, y_pred_mle)
    acc_map = accuracy_score(y_test, y_pred_map)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"MLE Accuracy: {acc_mle:.4f}")
    print(f"MAP Accuracy: {acc_map:.4f}")
    
    # Get scores (average log-likelihood)
    score_train_mle = model_mle.score(X_train_flat, y_train)
    score_test_mle = model_mle.score(X_test_flat, y_test)
    score_train_map = model_map.score(X_train_flat, y_train)
    score_test_map = model_map.score(X_test_flat, y_test)
    
    print(f"\nMLE - Training Score: {score_train_mle:.4f}")
    print(f"MLE - Test Score:     {score_test_mle:.4f}")
    print(f"MAP - Training Score: {score_train_map:.4f}")
    print(f"MAP - Test Score:     {score_test_map:.4f}")
    
    print("\nNote: Higher scores = better fit")
    print("If training score >> test score, we're overfitting!")
    print("=" * 60)