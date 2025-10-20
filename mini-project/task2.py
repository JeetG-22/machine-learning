import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class CategoricalNaiveBayes(BaseEstimator, ClassifierMixin):
    """
    Categorical Naive Bayes for EMNIST digit/letter classification.
    
    This is based on what we learned in Lecture 2 and 3 about:
    - Naive Bayes (assumes pixels are independent given the class)
    - MLE vs MAP (with vs without priors)
    
    Parameters:
    -----------
    alpha : float, default=1.0
        Prior for class probabilities (from Lecture 3 - Dirichlet)
        Think of it as "fake counts" we add to each class
    beta : float, default=1.0
        Prior for pixel probabilities (from Lecture 3 - Beta)
        Helps smooth pixel probabilities, prevents zeros
    method : str, default='MAP'
        'MLE' = no priors (Lecture 2)
        'MAP' = with priors (Lecture 3)
    """
    
    def __init__(self, alpha=1.0, beta=1.0, method='MAP'):
        self.alpha = alpha
        self.beta = beta
        self.method = method
        
    def fit(self, X, y):
        """
        Train the model on data.
        
        From Lecture 2: We're learning p(x, y) = p(x|y) * p(y)
        - p(y) = how common each class is
        - p(x|y) = what the images look like for each class
        """
        # Make sure data is flat (not 28x28 images)
        if X.ndim > 2:
            num_samples = X.shape[0]
            X = X.reshape(num_samples, -1)
        
        # Figure out what we're working with
        self.n_features_ = X.shape[1]  # Should be 784 (28*28)
        self.classes = np.unique(y)  # All the different classes
        self.n_classes = len(self.classes)  # Should be 47 for EMNIST
        
        # Step 1: Learn class probabilities
        # From Lecture 3: this is either MLE or MAP
        if self.method == 'MLE':
            self.class_log_prior_ = self.class_prior_mle(y)
        else:
            self.class_log_prior_ = self.class_prior_map(y)
        
        # Step 2: Learn pixel probabilities for each class
        # From Lecture 2: Naive Bayes assumes pixels are independent
        if self.method == 'MLE':
            self.feature_log_prob_ = self.pixel_prob_mle(X, y)
        else:
            self.feature_log_prob_ = self.pixel_prob_map(X, y)
        
        return self
    
    def class_prior_mle(self, y):
        """
        MLE: just count how often each class appears.
        
        From Lecture 2:
        pi_c = (# of times we see class c) / (total # of samples)
        """
        # Count samples in each class
        counts = np.zeros(self.n_classes)
        for i, c in enumerate(self.classes):
            counts[i] = np.sum(y == c)
        
        # Calculate probabilities (just divide by total)
        total = len(y)
        probs = counts / total
        
        # Return as log (more stable for computation)
        return np.log(probs)
    
    def class_prior_map(self, y):
        """
        MAP: like MLE but with a prior (smoothing).
        
        From Lecture 3:
        We start with a Dirichlet prior: Dir(alpha, alpha, ..., alpha)
        After seeing data, posterior is: Dir(alpha + N_0, alpha + N_1, ...)
        MAP estimate (the mode): pi_c = (N_c + alpha - 1) / (N + C*alpha - C)
        
        The alpha acts like "fake counts" we had before seeing any data
        """
        # Count samples in each class
        counts = np.zeros(self.n_classes)
        for i, c in enumerate(self.classes):
            counts[i] = np.sum(y == c)
        
        # Add the prior (alpha - 1 to each class)
        numerator = counts + self.alpha - 1
        # Total is: actual data + all the fake counts
        denominator = len(y) + self.n_classes * self.alpha - self.n_classes
        
        probs = numerator / denominator
        
        return np.log(probs)
    
    def pixel_prob_mle(self, X, y):
        """
        MLE for pixels: what's the probability each pixel is "on"?
        
        From Lecture 2 (Bernoulli MLE):
        theta = (# times pixel is 1) / (# total times we see that class)
        
        We do this for every pixel and every class
        """
        # Make array to store: (47 classes) x (784 pixels)
        pixel_probs = np.zeros((self.n_classes, self.n_features_))
        
        # For each class
        for i, c in enumerate(self.classes):
            # Get all images from this class
            X_class = X[y == c]
            
            # For each pixel, what fraction of the time is it "on"?
            # mean works because pixels are 0 or 1
            pixel_probs[i, :] = np.mean(X_class, axis=0)
            
            # Problem: if a pixel is ALWAYS off, we get 0, then log(0) = -inf
            # Solution: clip to very small values
            pixel_probs[i, :] = np.clip(pixel_probs[i, :], 1e-10, 1 - 1e-10)
        
        return np.log(pixel_probs)
    
    def pixel_prob_map(self, X, y):
        """
        MAP for pixels: MLE but with smoothing from Beta prior.
        
        From Lecture 3 (Beta-Bernoulli):
        Prior: theta ~ Beta(B, B)
        Posterior: theta | data ~ Beta(B + # of 1s, B + # of 0s)
        MAP estimate: theta = (# of 1s + B - 1) / (# total + 2B - 2)
        
        This prevents us from getting 0 probabilities!
        """
        # Same setup as MLE
        pixel_probs = np.zeros((self.n_classes, self.n_features_))
        
        for i, c in enumerate(self.classes):
            # Get images from this class
            X_class = X[y == c]
            num_in_class = X_class.shape[0]
            
            # Count how many times each pixel is "on" (= 1)
            ones_count = np.sum(X_class, axis=0)
            
            # Apply MAP formula
            # B - 1 is like adding "fake observations" before seeing data
            numerator = ones_count + self.beta - 1
            denominator = num_in_class + 2 * self.beta - 2
            
            pixel_probs[i, :] = numerator / denominator
            
            # Still clip just to be safe
            pixel_probs[i, :] = np.clip(pixel_probs[i, :], 1e-10, 1 - 1e-10)
        
        return np.log(pixel_probs)
    
    def predict(self, X):
        """
        Predict which class each test image belongs to.
        
        From Lecture 2: Use Bayes Rule
        Pick class c that maximizes: p(y=c|x) ∝ p(x|y=c) * p(y=c)
        
        In log space: log p(x,y=c) = log p(y=c) + log p(x|y=c)
        """
        # Flatten if needed
        if X.ndim > 2:
            num_samples = X.shape[0]
            X = X.reshape(num_samples, -1)
        
        # Calculate log probability for each class
        log_probs = self._joint_log_likelihood(X)
        
        # Pick the class with highest probability
        best_class_idx = np.argmax(log_probs, axis=1)
        return self.classes[best_class_idx]
    
    def _joint_log_likelihood(self, X):
        """
        Calculate log p(x, y=c) for every class c.
        
        From Lecture 2 (Naive Bayes):
        p(x, y=c) = p(y=c) * ∏ p(pixel_d | y=c) for all pixels d
        
        In log space (from Lecture 1 - avoid underflow):
        log p(x, y=c) = log p(y=c) + Σ log p(pixel_d | y=c)
        
        For each pixel:
        - If pixel is 1: add log(theta)
        - If pixel is 0: add log(1-theta)
        """
        num_samples = X.shape[0]
        log_probs = np.zeros((num_samples, self.n_classes))
        
        # For each class
        for c_idx in range(self.n_classes):
            # Start with: log p(y = c)
            log_probs[:, c_idx] = self.class_log_prior_[c_idx]
            
            # Now add: log p(x | y = c)
            # This is the Naive Bayes part - assume pixels independent
            
            # Get the pixel probabilities for this class
            log_theta = self.feature_log_prob_[c_idx, :]  # log P(pixel=1)
            log_one_minus_theta = np.log(1 - np.exp(log_theta))  # log P(pixel=0)
            
            # For each test sample
            for n in range(num_samples):
                # Add up contributions from all pixels
                # If pixel is 1, use log_theta; if 0, use log_one_minus_theta
                pixel_contribution = X[n] * log_theta + (1 - X[n]) * log_one_minus_theta
                log_probs[n, c_idx] += np.sum(pixel_contribution)
        
        return log_probs
    
    def score(self, X, y):
        """
        Calculate average log-likelihood of the data.
        
        From project: Score = (1/N) * Σ log p(x_n, y_n)
        
        Higher score = model fits data better
        """
        # Flatten if needed
        if X.ndim > 2:
            num_samples = X.shape[0]
            X = X.reshape(num_samples, -1)
        
        # Get probabilities for all classes
        log_probs = self._joint_log_likelihood(X)
        
        # For each sample, get probability of its TRUE class
        log_likelihoods = []
        for i, true_class in enumerate(y):
            # Find where this class is in our classes array
            where_class = np.where(self.classes == true_class)[0]
            
            if len(where_class) > 0:
                # We've seen this class before - use its probability
                class_idx = where_class[0]
                log_likelihoods.append(log_probs[i, class_idx])
            else:
                # We never saw this class in training!
                # From Lecture 3: assign very low (but not zero) probability
                log_likelihoods.append(-1000)
        
        # Return the average
        return np.mean(log_likelihoods)


# Testing the implementation
if __name__ == "__main__":
    print("=" * 60)
    print("Testing my Naive Bayes Implementation")
    print("=" * 60)
    
    # Load the data
    from emnist_project import X_train, X_test, y_train, y_test
    
    # Flatten the images
    X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(float)
    X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(float)
    
    print(f"\nDataset:")
    print(f"  Training: {X_train_flat.shape[0]} samples")
    print(f"  Testing: {X_test_flat.shape[0]} samples")
    print(f"  Features: {X_train_flat.shape[1]} pixels")
    print(f"  Classes: {len(np.unique(y_train))}")
    
    # Train MLE version (no priors)
    print("\n" + "-" * 60)
    print("Training MLE model...")
    model_mle = CategoricalNaiveBayes(alpha=1.0, beta=1.0, method='MLE')
    model_mle.fit(X_train_flat, y_train)
    print("Done!")
    
    # Train MAP version (with priors)
    print("\nTraining MAP model...")
    model_map = CategoricalNaiveBayes(alpha=1.0, beta=1.0, method='MAP')
    model_map.fit(X_train_flat, y_train)
    print("Done!")
    
    # Make predictions
    print("\nTesting predictions...")
    y_pred_mle = model_mle.predict(X_test_flat)
    y_pred_map = model_map.predict(X_test_flat)
    
    # Check accuracy
    from sklearn.metrics import accuracy_score
    acc_mle = accuracy_score(y_test, y_pred_mle)
    acc_map = accuracy_score(y_test, y_pred_map)
    
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"MLE Accuracy: {acc_mle:.4f}")
    print(f"MAP Accuracy: {acc_map:.4f}")
    
    # Get log-likelihood scores
    print("\nScoring models (higher = better)...")
    train_score_mle = model_mle.score(X_train_flat, y_train)
    test_score_mle = model_mle.score(X_test_flat, y_test)
    train_score_map = model_map.score(X_train_flat, y_train)
    test_score_map = model_map.score(X_test_flat, y_test)
    
    print(f"\nMLE:")
    print(f"  Training score:   {train_score_mle:.4f}")
    print(f"  Test score:       {test_score_mle:.4f}")
    print(f"  Gap (overfitting?): {train_score_mle - test_score_mle:.4f}")
    
    print(f"\nMAP:")
    print(f"  Training score:   {train_score_map:.4f}")
    print(f"  Test score:       {test_score_map:.4f}")
    print(f"  Gap (overfitting?): {train_score_map - test_score_map:.4f}")
    
    print("\nFrom Lecture 2: Big gap = overfitting!")
    print("From Lecture 3: MAP should have smaller gap than MLE")
    print("=" * 60)