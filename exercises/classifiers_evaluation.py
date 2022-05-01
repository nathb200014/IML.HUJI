import math

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from IMLearn.metrics.loss_functions import accuracy
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

from utils import decision_surface, custom

pio.templates.default = "simple_white"

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
        X, y = load_dataset("/Users/nathanaelbohbot/Documents/HUJI/IML/IML.HUJI/datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        def callback(perceptron : Perceptron, arr : np.ndarray, num : int):
            losses.append(perceptron._loss(X, y))
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

def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("/Users/nathanaelbohbot/Documents/HUJI/IML/IML.HUJI/datasets/" + f)

        # Fit models and predict over training set

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy

        lda = LDA()
        lda.fit(X, y)
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)

        fig = make_subplots(rows=1, cols=2, subplot_titles=
        ("GNB accuracy: " + str(accuracy(y, gnb.predict(X))), "LDA accuracy:  " + str(accuracy(y, lda.predict(X)))))

        lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-.4, .4])

        symbols = np.array(["circle-dot", "square", "triangle-right"])

        for i, m in enumerate([gnb, lda]):
            fig.add_traces([decision_surface(m._predict, lims[0], lims[1], showscale=False),
                            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                       marker=dict(symbol=symbols[y.astype(int)], color=y,
                                                   colorscale=[custom[0], custom[-1], custom[2]])
                                       ),
                            go.Scatter(x=m.mu_[:, 0], y=m.mu_[:, 1], mode='markers', marker=dict(color="black",
                                                                                                 symbol="x"),
                                       showlegend=False)],
                           rows=(i // 2) + 1, cols=(i % 2) + 1)

        ellipses_count = 0
        while ellipses_count < 3:
            fig.add_trace(
                get_ellipse(lda.mu_[ellipses_count], lda.cov_),
                row=1, col=2
            )
            fig.add_trace(
                get_ellipse(gnb.mu_[ellipses_count], np.diag(gnb.vars_[ellipses_count])),
                row=1, col=1
            )
            ellipses_count+=1
        fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
