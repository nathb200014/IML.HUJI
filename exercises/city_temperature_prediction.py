import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test
from IMLearn.metrics import mean_square_error
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    # preprocess
    df = df.dropna()
    df = df.drop_duplicates(subset=['Country','City','Date'], keep=False)
    # remove errors from remaining columns
    df = df[1 <= df["Month"]]
    df = df[df["Month"] <= 12]
    df = df[1 <= df["Day"]]
    df = df[df["Day"] <= 31]
    df = df[df["Temp"] > -25]
    # new category based on existing ones
    df['DayOfYear'] = df['Date'].dt.dayofyear
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data("/Users/nathb200014/Desktop/nb/IML.HUJI/datasets/City_Temperature.csv")
    # Question 2 - Exploring data for specific country
    df_cond = X.loc[X['Country'] == "Israel"]
    graph = px.scatter(df_cond, x='DayOfYear', y='Temp', color='Year',
                       title="average daily temperature change as function of the 'DayOfYear'",
                       labels={"x": "DayOfYear", "y": "Temperature"})
    graph.show()
    groups = df_cond.groupby('Month').Temp.std()
    index = []
    for i in range(1,13):
        index.append(i)
    graph2 = px.bar(groups, x=index, y='Temp',
                 title="Standard deviation of the daily temperatures for each month",
                 labels={"x": f"Month", "y": "Daily temperature's standard deviation"})
    graph2.show()


    # # Question 3 - Exploring differences between countries
    groups2 = X.groupby(["Country", 'Month'])["Temp"].agg(["mean", "std"])
    groups2 = groups2.reset_index()
    line = px.line(x=groups2.Month, y=groups2['mean'],
                  color=groups2.Country, error_y=groups2['std'],
                   title = "average monthly temperature with error bars color coded by the country",
                    labels = {"x": "Month", "y": "Average monthly temperature"})
    line.show()

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(df_cond['DayOfYear'], df_cond["Temp"], 0.75)
    Mse = []
    index_lst = []
    for k in range(1,11):
        p = PolynomialFitting(k)
        p._fit(train_X, train_y)
        mse = round(mean_square_error(test_y, p._predict(test_X)), 2)
        print("test error: ", mse)
        Mse.append(mse)
        index_lst.append(k)
    graph3 = px.bar(x=index_lst, y=Mse,
                    title="Test error recorded for each value of k",
                    labels={"x": "K", "y": "Mse"})
    graph3.show()

    #Question 5 - Evaluating fitted model on different countries
    model = PolynomialFitting(5) 
    model._fit(df_cond['DayOfYear'], df_cond["Temp"])
    countries = ['South Africa','The Netherlands', 'Jordan']
    Mse = []
    for country in countries:
        mse = round(mean_square_error(X[X.Country == country].Temp, model._predict(X[X.Country == country].DayOfYear))
                    , 2)
        Mse.append(mse)
    graph4 = px.bar(x=countries, y=Mse,
                 title="Model's error over each of the countries",
                 labels={"x": f"Countries", "y": "Mse"})
    graph4.show()

