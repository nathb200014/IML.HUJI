from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product

from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        best_loss = np.inf
        for sign, i in product([1, -1], range(X.shape[1])):
            curr_th, curr_loss = self._find_threshold(X[:, i], y, sign)
            if curr_loss < best_loss:
                self.threshold_ = curr_th
                self.j_ = i
                self.sign_ = sign
                best_loss = curr_loss

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.where(X[:, self.j_] >= self.threshold_, self.sign_, -self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # i = np.argsort(values)
        # values , labels = values[i], labels[i]
        # losses = []
        # # for label in labels:
        # #     if sign != label:
        # #         losses.append(losses[-1] - 1/ len(labels))
        # #     else:
        # #         losses.append(losses[-1] + 1/ len(labels))
        # for i in range(len(labels)):
        #     new_labels = np.array([-sign if j < i
        #                            else sign
        #                            for j in range(len(labels))])
        #     losses.append(np.sum(labels != new_labels) / len(labels))
        # losses = np.array(losses)
        # return values[np.argmin(losses)], losses[np.argmin(losses)]
        sort_idx = np.argsort(values)
        values, labels = values[sort_idx], labels[sort_idx]
        sign_y = np.where(labels == 0, 1, labels)
        min_err = np.abs(labels[np.sign(sign_y) != sign]).sum()  # smallest possible loss
        errors = np.cumsum(np.append(min_err, sign * labels[:-1]))
        best_index = np.argmin(errors)
        return values[best_index], errors[best_index]

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
