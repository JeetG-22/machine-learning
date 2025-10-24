import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from emnist_project import X_train, X_test, y_train, y_test
from sklearn.metrics import accuracy_score

class CategoricalNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0, beta=1.0, method='MAP'):
        self.alpha = alpha
        self.beta = beta
        self.method = method
        
    def fit(self, X, y):
        """
        Train the model on the data
        We learn p(x, y) = p(x|y) * p(y)
        """
        # flatten data if needed
        if X.ndim > 2:
            num_samples = X.shape[0]
            X = X.reshape(num_samples, -1)
        
        self.n_features = X.shape[1]  # 784 for 28x28 images
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)  # should be 47
        
        # learn class probabilities
        if self.method == 'MLE':
            self.class_log_prior_ = self.class_prior_mle(y)
        else:
            self.class_log_prior_ = self.class_prior_map(y)
        
        # learn pixel probabilities
        if self.method == 'MLE':
            self.features_log_prob = self.pixel_prob_mle(X, y)
        else:
            self.features_log_prob = self.pixel_prob_map(X, y)
        
        return self
    
    def class_prior_mle(self, y):
        """MLE for class probabilities - just count frequencies"""
        counts = np.zeros(self.n_classes)
        # print(y)
        # print(self.classes)
        for i, c in enumerate(self.classes):
            counts[i] = np.sum(y == c)
        
        # convert to probabilities
        total = len(y)
        probs = counts / total
        
        # use log for numerical stability
        return np.log(probs)
    
    def class_prior_map(self, y):
        """MAP for class probabilities - adds smoothing with Dirichlet prior"""
        counts = np.zeros(self.n_classes)
        for i, c in enumerate(self.classes):
            counts[i] = np.sum(y == c)
        
        # add the prior (like adding fake counts)
        numerator = counts + self.alpha - 1
        denominator = len(y) + self.n_classes * self.alpha - self.n_classes
        
        probs = numerator / denominator
        
        return np.log(probs)
    
    def pixel_prob_mle(self, X, y):
        """MLE for pixel probabilities"""
        pixel_probs = np.zeros((self.n_classes, self.n_features))
        
        for i, c in enumerate(self.classes):
            # get all images from this class
            X_class = X[y == c]
            
            # what fraction of time is each pixel "on"?
            pixel_probs[i, :] = np.mean(X_class, axis=0)
            
            # avoid log(0) problems
            pixel_probs[i, :] = np.clip(pixel_probs[i, :], 1e-10, 1 - 1e-10)
        
        return np.log(pixel_probs)
    
    def pixel_prob_map(self, X, y):
        """MAP for pixel probabilities - uses Beta prior for smoothing"""
        pixel_probs = np.zeros((self.n_classes, self.n_features))
        
        for i, c in enumerate(self.classes):
            X_class = X[y == c]
            num_in_class = X_class.shape[0]
            
            # count how many times each pixel is 1
            ones_count = np.sum(X_class, axis=0)
            
            # apply MAP formula with Beta prior
            numerator = ones_count + self.beta - 1
            denominator = num_in_class + 2 * self.beta - 2
            
            pixel_probs[i, :] = numerator / denominator
            
            # clip just to be safe
            pixel_probs[i, :] = np.clip(pixel_probs[i, :], 1e-10, 1 - 1e-10)
        
        return np.log(pixel_probs)
    
    def predict(self, X):
        """Predict class for each sample"""
        if X.ndim > 2:
            num_samples = X.shape[0]
            X = X.reshape(num_samples, -1)
        
        # get log probabilities for all classes
        log_probs = self.join_log_likelihood(X)
        
        # pick the best class
        best_class_idx = np.argmax(log_probs, axis=1)
        return self.classes[best_class_idx]
    
    def join_log_likelihood(self, X):
        """
        Calculate log p(x, y=c) for every class
        Using Naive Bayes assumption that pixels are independent
        """
        num_samples = X.shape[0]
        log_probs = np.zeros((num_samples, self.n_classes))
        
        for c_idx in range(self.n_classes):
            # start with class prior
            log_probs[:, c_idx] = self.class_log_prior_[c_idx]
            
            # add pixel contributions
            log_theta = self.features_log_prob[c_idx, :]
            log_one_minus_theta = np.log(1 - np.exp(log_theta))
            
            # for each sample
            for n in range(num_samples):
                # if pixel is 1, add log(theta); if 0, add log(1-theta)
                pixel_contribution = X[n] * log_theta + (1 - X[n]) * log_one_minus_theta
                log_probs[n, c_idx] += np.sum(pixel_contribution)
        
        return log_probs
    
    def score(self, X, y):
        """Calculate average log-likelihood"""
        if X.ndim > 2:
            num_samples = X.shape[0]
            X = X.reshape(num_samples, -1)
        
        log_probs = self.join_log_likelihood(X)
        
        # get probability of true class for each sample
        log_likelihoods = []
        for i, true_class in enumerate(y):
            where_class = np.where(self.classes == true_class)[0]
            
            if len(where_class) > 0:
                class_idx = where_class[0]
                log_likelihoods.append(log_probs[i, class_idx])
            else:
                # unseen class - give it very low probability
                log_likelihoods.append(-1000)
        
        return np.mean(log_likelihoods)


# Test the implementation
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Naive Bayes Implementation")
    print("=" * 60)
    
    # flatten the data
    X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(float)
    X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(float)
    
    print(f"\nDataset info:")
    print(f"  Training samples: {X_train_flat.shape[0]}")
    print(f"  Test samples: {X_test_flat.shape[0]}")
    print(f"  Features (pixels): {X_train_flat.shape[1]}")
    print(f"  Number of classes: {len(np.unique(y_train))}")
    
    # train MLE model
    print("\n" + "-" * 60)
    print("Training MLE model...")
    model_mle = CategoricalNaiveBayes(alpha=1.0, beta=1.0, method='MLE')
    model_mle.fit(X_train_flat, y_train)
    print("Done!")
    
    # train MAP model
    print("\nTraining MAP model...")
    model_map = CategoricalNaiveBayes(alpha=1.0, beta=1.0, method='MAP')
    model_map.fit(X_train_flat, y_train)
    print("Done!")
    
    # make predictions
    print("\nMaking predictions...")
    y_pred_mle = model_mle.predict(X_test_flat)
    y_pred_map = model_map.predict(X_test_flat)
    
    # check accuracy
    acc_mle = accuracy_score(y_test, y_pred_mle)
    acc_map = accuracy_score(y_test, y_pred_map)
    
    print("\n")
    print("Results:")
    print("=" * 60)
    print(f"MLE Accuracy: {acc_mle}")
    print(f"MAP Accuracy: {acc_map}")
    
    # get scores
    print("\nCalculating log-likelihood scores...")
    train_score_mle = model_mle.score(X_train_flat, y_train)
    test_score_mle = model_mle.score(X_test_flat, y_test)
    train_score_map = model_map.score(X_train_flat, y_train)
    test_score_map = model_map.score(X_test_flat, y_test)
    
    print(f"\nMLE:")
    print(f"  Training score:   {train_score_mle}")
    print(f"  Test score:       {test_score_mle}")
    print(f"  Gap: {train_score_mle - test_score_mle}")
    
    print(f"\nMAP:")
    print(f"  Training score:   {train_score_map}")
    print(f"  Test score:       {test_score_map}")
    print(f"  Gap: {train_score_map - test_score_map}")
    
    print("=" * 60)