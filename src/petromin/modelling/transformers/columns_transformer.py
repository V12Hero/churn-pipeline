import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select columns from a DataFrame.

    This transformer selects columns from a pandas DataFrame. It's useful as
    part of a scikit-learn pipeline to apply transformations to specific columns.

    Attributes:
        columns (list of str): The list of column names to select.

    Examples:
        >>> import pandas as pd
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.preprocessing import StandardScaler

        # Example DataFrame
        >>> data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6], 'feature3': [7, 8, 9]}
        >>> df = pd.DataFrame(data)

        # Using ColumnSelector to select specific columns
        >>> selector = ColumnSelector(columns=['feature1', 'feature2'])
        >>> selected_df = selector.transform(df)

        # Integrating ColumnSelector into a pipeline
        >>> pipeline = Pipeline([
        ...     ('selector', ColumnSelector(columns=['feature1', 'feature2'])),
        ...     ('scaler', StandardScaler())
        ... ])
        >>> pipeline.fit_transform(df)

    """

    def __init__(self, columns):
        """
        Args:
            columns (list of str): The list of column names to select.
        """
        self.columns = columns

    def fit(self, X, y=None):
        """Fits the transformer on the data.

        This method doesn't do anything except setting the _fitted attribute to True.
        It's here to comply with the scikit-learn transformer interface.

        Args:
            X (pd.DataFrame): The data to fit on.
            y (ignored): Not used, present here for API consistency by convention.

        Returns:
            self (ColumnSelector): The fitted transformer.
        """
        self._fitted = True
        return self

    def transform(self, X):
        """Transform the DataFrame by selecting specified columns.

        Args:
            X (pd.DataFrame): The data to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame with only the selected columns.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "X must be a pandas DataFrame in order to preserve column and index information"
            )

        X_transformed = X[self.columns]

        return X_transformed
