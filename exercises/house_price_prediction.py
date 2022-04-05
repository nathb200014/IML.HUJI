import os

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset
    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    #preprocess
    df = df.dropna()
    df = df.drop('id', axis=1)
    df = df.drop('date', axis=1)
    #remove errors from remaining columns
    col_names = ["price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot"]
    for s in col_names:
        df = df[df[s] > 0]
    #categorical with no logical order
    pd.get_dummies(df.zipcode)
    #new category based on existing ones
    df['sqft_overall'] = df.apply(lambda row: row.price / (row.sqft_living + (1/3)*row.sqft_basement),
                                  axis=1)
    price = df.price
    df = df.drop(["price"], axis=1)
    return df, price

def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    y : array-like of shape (n_samples, )
        Response vector to evaluate against
    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    X = X.assign(price = y)
    cov = X.cov()
    sd_y = X['price'].std()
    for i in X.columns:
        sd_x = X[i].std()
        denominator = 1 / (sd_x * sd_y)
        corr = cov[i]['price'] * denominator
        # py.title("Pearson correlation under specific model")
        # py.xlabel("feature")
        # py.ylabel("response")
        # py.scatter(X[i], y)
        # py.show()
        graph = go.Figure([go.Scatter(x=X[i], y=y, mode='markers')],
                          layout=go.Layout(title=r"$\text{Correlation between " + str(i) +
                                                 " and price ---" + "pearson correlation: " + str(corr) + " ""}$",
                                           xaxis_title=r"$\text{" + str(i) + "}$",
                                           yaxis_title=r"$\text{price}$"))
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        graph.write_image(output_path + "/" + str(i) + ".png")



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("/cs/usr/nathb200014/PycharmProjects/IML.HUJI/datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, "bohbot") #todo ajouter path
    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    loss_mean = sd_arr = np.zeros(101)
    loss = None
    for p in range(10, 101):
        for time in range(10):
            loss = np.zeros(10)
            sampled_x = train_X.sample(frac=p/100)
            sampled_y = train_y[train_X.index] #todo cest comme ca ?
            model = LinearRegression()
            model._fit(sampled_x.values, sampled_y.values)
            loss_num = model._loss(test_X.values, test_y.values)
            np.append(loss, loss_num)
        np.append(loss_mean, np.mean(loss))
        np.append(sd_arr, np.std(loss))
    # #todo ajoute instructions graph