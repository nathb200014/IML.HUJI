import math
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from ...metrics import loss_functions


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier
        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`
        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`
        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`
        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        self.mu_ = []
        self.pi_ = np.zeros(self.classes_.size)
        self.vars_ = []
        i = 0
        for k in self.classes_:
            self.mu_.append(np.mean(X[np.where(y == k)], axis=0))
            self.pi_[i] = (1/X.shape[0])*X[np.where(y == k)].size
            self.vars_.append(np.var(X[np.where(y == k)], axis=0))
            i += 1
        self.mu_ = np.array(self.mu_)
        self.vars_ = np.array(self.vars_)
        self.fitted_ = True


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for
        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return np.argmax(self.likelihood(X), axis=1)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.
        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes
        """
        l = []
        for k in range(self.classes_.size):
            cov = np.diag(self.vars_[k])
            part1 = 1 / (np.sqrt(math.pow(2 * np.pi, X.shape[1]) * np.linalg.det(cov)))
            lst = []
            for x in X:
                mat = x - self.mu_[k]
                part2 = (-1 / 2) * np.matmul(np.matmul(mat, np.linalg.inv(cov)), mat.T)
                lst.append(part1 * np.exp(part2) * self.pi_[k])
            l.append(lst)
        return np.array(l).T

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples, )
            True labels of test samples
        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """

        return loss_functions.misclassification_error(y, self._predict(X))

