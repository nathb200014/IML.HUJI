import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics import accuracy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaBoost = AdaBoost(DecisionStump, n_learners)
    adaBoost.fit(train_X, train_y)
    training_error = []
    test_error = []
    for max_t in range(1,n_learners):
        test_error.append(adaBoost.partial_loss(test_X, test_y, max_t))
        training_error.append(adaBoost.partial_loss(train_X, train_y, max_t))
    fig = go.Figure([
        go.Scatter(x=list(range(1, n_learners)), y=training_error, name='train error'),
        go.Scatter(x=list(range(1, n_learners)), y=test_error, name='test error')])
    fig.update_layout(title="Training and test errors as a function of the number of fitted learners",
                      xaxis=dict(title="Training and test errors"), yaxis=dict(title="number of fitted learners"))
    fig.show()
    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(2, 2, subplot_titles=[f"{t} classifiers" for t in T])
    for i, T in enumerate(T):
        fig.add_traces([decision_surface(lambda x: adaBoost.partial_predict(x, T), lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y,
                                               colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig.update_layout(title=rf"$\textbf{{Decision boundary received by using the ensemble at different iterations}}$",
                      margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()
    # Question 3: Decision surface of best performing ensemble
    fig = go.Figure()
    fig.add_traces([decision_surface(lambda x: adaBoost.partial_predict(x, np.argmin(test_error) + 1), lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=test_y,colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1)))])
    fig.update_layout(title=f"Best decision boundary with noise = {noise}, ensemble size ="
                            f" {np.argmin(test_error) + 1} and accuracy = "
                            f"{accuracy(adaBoost.partial_predict(test_X, T), test_y)}")
    fig.show()
    # Question 4: Decision surface with weighted samples
    fig = go.Figure()
    fig.add_traces([decision_surface(lambda X: adaBoost.predict(X), lims[0], lims[1], showscale=False),
                    go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=train_y, colorscale=[custom[0], custom[-1]],
                                           size=(adaBoost.D_ / np.max(adaBoost.D_))*5,
                                           line=dict(color="black", width=1)))])
    fig.update_layout(title=f"Weighted train decision boundaries with noise = {noise}")
    fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
