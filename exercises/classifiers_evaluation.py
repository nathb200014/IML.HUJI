import math

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from IMLearn.metrics.loss_functions import accuracy
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
pio.templates.default = "simple_white"

import matplotlib.pyplot as plt


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    return np.load(filename)[:, :2], np.load(filename)[:, 2]

def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("/cs/usr/nathb200014/IML.HUJI/datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        def callback(perceptron : Perceptron, arr : np.ndarray, num : int):
            losses.append(perceptron._loss(X, y))

        print(losses  )
        # Plot figure
        a = Perceptron(callback=callback)
        a._fit(X, y)
        line = px.line(x=range(len(losses)), y=losses,
                       title="Evolution of the losses during the iterations",
                       labels={"x": "Iterations", "y": "Losses"})
        line.show()

def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = math.atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * math.pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")

# def compare_gaussian_classifiers():
#     """
#     Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
#     """
#     for f in ["gaussian1.npy", "gaussian2.npy"]:
#         # Load dataset
#         X, y = load_dataset("/Users/nathanaelbohbot/Documents/HUJI/IML/IML.HUJI/datasets/" + f)
#
#         # Fit models and predict over training set
#         lda = LDA()
#         lda.fit(X, y)
#         gauss = GaussianNaiveBayes()
#         gauss.fit(X, y)
#         gnb_accuracy = accuracy(y, gauss.predict(X))
#         # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
#         # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
#         models = [lda, gauss]
#
#         fig = make_subplots(rows=1, cols=2,
#                             subplot_titles=[f"{t} "
#                                             f"accuracy: "
#                                             f"{accuracy(y, m._predict(X))}"
#                                             for m, t in models],
#                             horizontal_spacing=0.01, vertical_spacing=.03)
#         # fig.update_layout(title="Data from {f} dataset".format(f=f[:-4]), margin=dict(t=100))
#         symbols = np.array(["circle", "square", "triangle-up"])
#         for i, m in enumerate(models):
#             fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
#                                      marker=dict(size=10, color=m.predict(X), symbol=symbols[y],
#                                                  line=dict(color="black", width=1))),
#                           row=(i // 2) + 1, col=(i % 2) + 1)
#
#             fig.add_trace(go.Scatter(x=m.mu_[:, 0], y=m.mu_[:, 1], mode="markers", showlegend=False,
#                                      marker=dict(size=30, color="black", symbol="x")),
#                           row=(i // 2) + 1, col=(i % 2) + 1)
#
#         # fig.update_layout(showlegend=False)
#         for i in range(len(lda.classes_)):
#             fig.add_trace(get_ellipse(lda.mu_[i], lda.cov_), row=1, col=2)
#         for i in range(len(gauss.classes_)):
#             fig.add_trace(get_ellipse(gauss.mu_[i], np.diag(gauss.vars_[i])), row=1, col=1)
#
#         fig.show()
def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    from IMLearn.metrics import accuracy

    for f in ["gaussian1.npy", "gaussian2.npy"]:
        path = r"../datasets/{f}".format(f=f)
        X,y = load_dataset(path)

        # Fit models and predict over training set
        #LDA
        lda = LDA()
        lda.fit(X,y)
        y_pred = lda.predict(X)
        lda_accuracy = accuracy(y,y_pred)

        #GNB
        gnb = GaussianNaiveBayes()
        gnb.fit(X,y)
        gnb_accuracy = accuracy(y, gnb.predict(X))

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy

        # Create subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Gaussian Naive Bayes Predictions. Accuracy: {gnb_a}".format(gnb_a=gnb_accuracy), "LDA Predictions. Accuracy: {lda_a}".format(lda_a=lda_accuracy)))
        fig.update_layout(title="Data from {f} dataset".format(f=f[:-4]), margin=dict(t=100))

        # Add traces for data-points setting symbols and colors
        symbols = np.array(["circle", "triangle-up", "square"])

        for i, m in enumerate([gnb, lda]):
            fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                            marker=dict(size=10, color=m.predict(X), symbol=symbols[y],
                            line=dict(color="black", width=1))),
                            row=(i // 2) + 1, col=(i % 2) + 1)

            # Add `X` dots specifying fitted Gaussians' means
            fig.add_trace(go.Scatter(x=m.mu_[:, 0], y=m.mu_[:, 1], mode="markers",showlegend=False,
                                     marker=dict(size=30, color="black", symbol="x")),
                          row=(i // 2) + 1, col=(i % 2) + 1)

        fig.update_layout(showlegend=False)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(len(lda.classes_)):
            fig.add_trace(get_ellipse(lda.mu_[i], lda.cov_), row=1, col=2)
        for i in range(len(gnb.classes_)):
            fig.add_trace(get_ellipse(gnb.mu_[i], np.diag(gnb.vars_[i])), row=1, col=1)

        fig.show()
if __name__ == '__main__':
    np.random.seed(0)
    #run_perceptron()
    compare_gaussian_classifiers()
