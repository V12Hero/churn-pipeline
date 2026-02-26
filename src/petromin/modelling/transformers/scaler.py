import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from petromin.python_utils.load.object_inyection import load_object


class ColumnsPreserverScaler(BaseEstimator, TransformerMixin):
    """
    A custom transformer that applies a specified scaler to a pandas DataFrame while preserving
    column names and indexes.

    Args:
        scaler_params (dict or str): A dictionary containing the class name and optional
        keyword arguments for the scaler object, or a string specifying the path
        to a saved scaler object. Default is the MinMaxScaler from scikit-learn.

    Attributes:
        columns (list): The list of column names from the input DataFrame.
        index (list): The list of index values from the input DataFrame.

    Methods:
        fit(X):
            Fits the specified scaler to the input data and stores column names and index values for later use.

        transform(X):
            Transforms the input data using the previously fitted scaler,
            retaining the original column names and indexes.

    Notes:
        - This transformer is designed to work with pandas DataFrames and is intended to preserve
        column names and indexes.
        - The `scaler_params` parameter can specify either a class name with optional keyword arguments or the path
            to a saved scaler object.
        - It is essential to use a scaler compatible with the input data to maintain meaningful transformations.

    Example:
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from my_module import ColumnIndexPreserverScaler

        # Sample DataFrame
        data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        df = pd.DataFrame(data)

        # Create a ColumnIndexPreserverScaler with a StandardScaler
        scaler = ColumnIndexPreserverScaler(scaler_params={"class": "sklearn.preprocessing.StandardScaler", "kwargs": None})
        scaler.fit(df)

        # Transform the DataFrame while preserving column names and indexes
        scaled_df = scaler.transform(df)
    """

    DEFAULT_PARAMS = {"class": "sklearn.preprocessing.MinMaxScaler", "kwargs": None}

    def __init__(self, scaler_params=DEFAULT_PARAMS):
        self.scaler_params = scaler_params
        self.scaler = load_object(self.scaler_params)

    def fit(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "X must be a pandas DataFrame in order to preserve column and index information"
            )

        self.columns = list(X.columns)
        self.index = list(X.index)
        self.scaler = self.scaler.fit(X)
        self.is_fitted = True
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            msg = (
                "X must be a pandas DataFrame in order to preserve "
                f"column and index information and must contains the following columns {self.columns}"
            )
            raise KeyError(msg)

        X = pd.DataFrame(
            self.scaler.transform(X[self.columns]), index=X.index, columns=self.columns
        )

        return X
