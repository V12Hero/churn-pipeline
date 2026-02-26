import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import BallTree


class BallTreeIndexer(BaseEstimator, ClusterMixin):
    """
    Ball Tree Indexer for efficient nearest neighbor search in high-dimensional data.

    This class implements a Ball Tree-based indexer that can efficiently find the nearest neighbors
    in a high-dimensional dataset. It is compatible with Scikit-learn's TransformerMixin and ClusterMixin
    interfaces.

    Args:
        None

    Attributes:
        None

    Methods:
        fit(X, y=None):
            Fit the Ball Tree indexer to the input data.

        predict(X):
            Find nearest neighbors in the fitted Ball Tree index.
    """

    def __init__(
        self,
    ):
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        """
        Fit the Ball Tree indexer to the input data.

        Args:
            X (pd.DataFrame): The input data for fitting the Ball Tree index.
            y (None, optional): Ignored. Placeholder for compatibility with Scikit-learn.

        Returns:
            self (BallTreeIndexer): The fitted Ball Tree indexer.

        """
        self.k = len(X)
        X_normalized = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns, index=X.index)
        self.ball_tree = BallTree(X_normalized)
        self.is_fitted = True
        self.features_in_ = list(X.columns)
        self.X = X
        self.X_normalized = X_normalized
        return self

    def predict(self, X: pd.DataFrame):
        """
        Find nearest neighbors in the fitted Ball Tree index.

        Args:
            X (pd.DataFrame): The input data for finding nearest neighbors.

        Returns:
            pd.DataFrame: A DataFrame containing information about the nearest neighbors.

        """
        if len(X) > 1:
            raise ValueError("You should query one data point at a time.")

        X_normalized = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        self.distances, self.indexes = self.ball_tree.query(X_normalized, k=self.k)
        df_dist = pd.DataFrame(
            [self.indexes[0], self.distances[0]], index=["indexes", "distances"]
        ).T
        df_dist["indexes"] = df_dist["indexes"].apply(int)
        df_dist["store_proximity"] = self.X.index[df_dist["indexes"].to_list()]
        df_dist = (
            df_dist.reset_index()
            .rename(columns={"index": "ranking"})
            .drop(columns=["indexes"])
            .set_index("store_proximity")
        )
        df_dist = df_dist.merge(self.X, left_index=True, right_index=True)
        return df_dist
