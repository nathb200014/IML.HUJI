from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from IMLearn.metrics import loss_functions


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier
    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`
    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`
    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`
    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`
    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.
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
        i = 0
        for k in self.classes_:
            self.mu_.append(np.mean(X[np.where(y == k)], axis=0))
            self.pi_[i] = (1/X.shape[0])*X[np.where(y == k)].size
            i += 1
        self.mu_ = np.array(self.mu_)
        for j in range(self.classes_.size):
            first_part = X[np.where(y == self.classes_[j])] - self.mu_[j]
            if self.cov_ is None:
                self.cov_ = np.matmul(first_part.T, first_part)
            else:
                self.cov_ += np.matmul(first_part.T, first_part)
        self.cov_ = (1/(X.shape[0] - self.classes_.size))*self.cov_
        self._cov_inv = np.linalg.inv(self.cov_)
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
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        l = []
        for x in range(X.shape[0]):
            arr = []
            for k in range(self.classes_.size):
                a = np.matmul(np.matmul(X[x].T, self._cov_inv), self.mu_[k].T)
                b = (-1 / 2) * np.matmul(np.matmul(self.mu_[k], self._cov_inv), self.mu_[k].T)
                arr.append(np.log(self.pi_[k]) + a + b)
            l.append(arr)
        return np.array(l)


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
