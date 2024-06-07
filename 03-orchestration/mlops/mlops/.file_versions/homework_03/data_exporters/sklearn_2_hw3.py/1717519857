from typing import Callable, Dict, Tuple, Union

from pandas import Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from mlops.utils.models.sklearn import load_class, train_model

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def train(
    settings: Tuple[
        csr_matrix,
        Series,
        BaseEstimator,
    ],
    **kwargs,
) -> Tuple[BaseEstimator, Dict[str, str]]:
    X, y, dv = settings

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    print(f'Train RMSE: {mean_squared_error(y, y_pred, squared=False)}')

    return model, dv
    