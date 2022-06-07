from __future__ import annotations
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

from ...metrics import mean_square_error
from ...utils import split_train_test


class RidgeRegression(BaseEstimator):
    """
    Ridge Regression Estimator

    Solving Ridge Regression optimization problem
    """

    def __init__(self, lam: float, include_intercept: bool = True) -> RidgeRegression:
        """
        Initialize a ridge regression model

        Parameters
        ----------
        lam: float
            Regularization parameter to be used when fitting a model

        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """


        """
        Initialize a ridge regression model
        :param lam: scalar value of regularization parameter
        """
        super().__init__()
        self.coefs_ = None
        self.include_intercept_ = include_intercept
        self.lam_ = lam

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Ridge regression model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        if self.include_intercept_:
            b = np.ones((X.shape[0], 1))
            X = np.hstack((b, X))
        d = X.shape[1] #todo mettre 0 ca marche ?
        first_part = np.matmul(X.T, X) + self.lam_ * np.identity(d)
        self.coefs_ = np.matmul(np.matmul(first_part, X.T), y)

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
        if self.include_intercept_:
            b = np.ones((X.shape[0], 1))
            X = np.hstack((b, X))
        return np.matmul(X, self.coefs_)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        return mean_square_error(y, self._predict(X))


class lol:
    from __future__ import annotations
    import IMLearn.learners
    import math
    from copy import deepcopy
    from typing import Tuple, Callable
    import numpy as np
    import pandas as pd

    from IMLearn import BaseEstimator
    from IMLearn.learners.regressors import PolynomialFitting

    def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                       scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
        """
        Evaluate metric by cross-validation for given estimator

        Parameters
        ----------
        estimator: BaseEstimator
            Initialized estimator to use for fitting the data

        X: ndarray of shape (n_samples, n_features)
           Input data to fit

        y: ndarray of shape (n_samples, )
           Responses of input data to fit to

        scoring: Callable[[np.ndarray, np.ndarray, ...], float]
            Callable to use for evaluating the performance of the cross-validated model.
            When called, the scoring function receives the true- and predicted values for each sample
            and potentially additional arguments. The function returns the score for given input.

        cv: int
            Specify the number of folds.

        Returns
        -------
        train_score: float
            Average train score over folds

        validation_score: float
            Average validation score over folds
        """
        X_split, Y_split = np.array_split(X, cv), np.array_split(y, cv)
        train = 0
        valid = 0
        for j in range(cv):
            train_X = np.concatenate(X_split[j][:j] + X_split[j][j+1:])
            train_Y = np.concatenate(Y_split[j][:j] + Y_split[j][j+1:])
            estimator.fit(train_X, train_Y)
            train += scoring(estimator.predict(train_X), train_Y)
            valid += scoring(estimator.predict(X_split[j]), Y_split[j])
        valid /= cv
        train /= cv
        return train, valid

class lol2:
    from __future__ import annotations

    from utils import *
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    output_path = "C:/Users/adjed/OneDrive/Bureau/IML/ex5/"

    def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
        """
        Simulate data from a polynomial model and use cross-validation to select the best fitting degree

        Parameters
        ----------
        n_samples: int, default=100
            Number of samples to generate

        noise: float, default = 5
            Noise level to simulate in responses
        """
        # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
        # and split into training- and testing portions
        X = np.linspace(-1.2, 2, n_samples)
        f = (X + 3)*(X + 2)*(X + 1)*(X - 1)*(X - 2)
        eps = np.random.normal(0, noise, n_samples)
        X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(X), pd.Series(f + eps), 2/3)
        fig = go.Figure([go.Scatter(x=X, y=f(X), mode="markers", name="Data without any noise"),
                         go.Scatter(x=X_train.to_numpy().reshape(X_train.shape[0]), y=y_train.to_numpy(),
                                    mode="markers", name="train"),
                         go.Scatter(x=X_test.to_numpy().reshape(X_test.shape[0]), y=y_test.to_numpy(), mode="markers",
                                    name="test")]).update_layout(title=dict(text=f'Data with noise of {noise}')).show()

        # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
        valid_lst = []
        train_lst = []
        for j in range(11):
            p = PolynomialFitting(j)
            train, valid = cross_validate(p, X_train, y_train, mean_square_error)
            train_lst.append(train)
            valid_lst.append(valid)
        go.Figure([go.Scatter(y=train_lst, x=list(range(11)), mode='lines', name="train"),
                   go.Scatter(y=valid_lst, x=list(range(11)), mode='lines', name="validation")],
                  layout=go.Layout(title="cross validation for polynomial fitting with several degrees",
                                   height=650, xaxis_title="polynomial degree",  yaxis_title="train loss")).show()
        # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
        best_val = np.argmin(np.array(valid_lst))
        best_p = PolynomialFitting(best_val).fit(X_train, y_train)
        test_error = round(mean_square_error(y_test, best_p.predict(X_test)), 2)
        print(f"the best value of k is {best_val} and the test error of the fitted model is {test_error}")

    def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
        """
        Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
        values for Ridge and Lasso regressions

        Parameters
        ----------
        n_samples: int, default=50
            Number of samples to generate

        n_evaluations: int, default = 500
            Number of regularization parameter values to evaluate for each of the algorithms
        """
        # Question 6 - Load diabetes dataset and split into training and testing portions
        X, y = datasets.load_diabetes(return_X_y=True)
        X_train, X_test, y_train, y_test = X[:n_samples], X[n_samples:], y[:n_samples], y[n_samples:]

        # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
        ridge_valid = []
        ridge_train = []
        lasso_valid = []
        lasso_train = []
        a = np.linspace(0, 1.1, n_evaluations)#todo changer valeurs
        for val in a:
            ridge = RidgeRegression(val)
            lasso = Lasso(alpha=val)
            train_r, valid_r = cross_validate(ridge, X_train, y_train, mean_square_error)
            train_l, valid_l = cross_validate(lasso, X_train, y_train, mean_square_error)
            ridge_train.append(train_r)
            ridge_valid.append(valid_r)
            lasso_train.append(train_l)
            lasso_valid.append(valid_l)
        lists_train = [ridge_train, lasso_train]
        lists_valid = [ridge_valid, lasso_valid]
        for i,j in enumerate(["Ridge", "Lasso"]):
            go.Figure([go.Scatter(x=a, y=lists_train[i], mode='lines', name="Train Error"),
                       go.Scatter(x=a, y=lists_valid[i], mode='lines', name="Validation Error")],
                      layout=go.Layout(title=f"cross validation for {j} Regularization with several lambdas values",
                                       height=650, xaxis_title="Lambda ", yaxis_title="Error")).show()

        # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
        lst1 = [ridge_valid, lasso_valid]
        lst2 = [RidgeRegression, Lasso, LinearRegression]
        lst3 = ["Ridge", "Lasso", "Linear regression"]
        lst4 = []
        for i in range(len(lst2)):
            if i == 2:
                lst4.append(lst2[i]().fit(X_train, y_train))
                break
            best = a[np.argmin(np.array(lst1[i]))]
            print(f"Best lambda for {lst3[i]} Regularization is {best}")
            lst4.append(lst2[i](best).fit(X_train, y_train))
        for j in range(len(lst2)):
            c = lst4[j].loss(X_test, y_test) if j != 1 else mean_square_error(lst4[j].predict(X_test), y_test)
            print(f"Test error of the {lst3[j]} model is: {c}")


    if __name__ == '__main__':
        np.random.seed(0)
        # q1, q2, q3
        select_polynomial_degree()
        # q4
        # select_polynomial_degree(noise=0)
        # q5
        # select_polynomial_degree(n_samples=1500, noise=10)
        # q7, q8
        # select_regularization_parameter()
