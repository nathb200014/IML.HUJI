import math

import numpy as np
from ..base import BaseEstimator
from typing import Callable, NoReturn

from ..metrics import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.models_ = []
        self.weights_ = []
        self.D_ = np.ones(X.shape[0]) / X.shape[0]
        D = np.zeros(X.shape[0])
        for t in range(self.iterations_):
            h = self.wl_()
            h._fit(X, y*self.D_)
            self.models_.append(h)
            epsilon = denominator = 0
            for i in range(X.shape[0]):
                bool = (y[i] != h._predict(X[i]))
                epsilon += self.D_[i]*bool
            w = (1/2)*np.log((1/epsilon)-1)
            self.weights_.append(w)
            for i in range(X.shape[0]):
                numerator = self.D_[i]*np.exp(-w*y[i]*h._predict(X[i]))
                for j in range(X.shape[0]):
                    denominator += self.D_[j]*np.exp(-w*y[j]*h._predict(X[j]))
                D[i] = numerator/denominator
            self.D_ = D

    def _predict(self, X):
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

        sum = 0
        for i in range(self.iterations_):
            sum += self.weights_[i]*self.models_[i]._predict(X)
        return np.sign(sum) #todo pas scalaire mais vecteur

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

        return misclassification_error(y, self._predict(X))

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        prev = self.iterations_
        self.iterations_ = T
        pred = self._predict(X)
        self.iterations_ = prev
        return pred

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """

        return misclassification_error(y, self.partial_predict(X, T))
