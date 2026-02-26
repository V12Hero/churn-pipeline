import logging

from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class ShapeLogger(BaseEstimator, TransformerMixin):
    """Transformer for logging the shape of data during transformations.

    This transformer is used for logging the shape of the data after a transformation
    in a machine learning pipeline. It wraps another transformer and logs the shape of
    the output of the transformer's transform method.

    Attributes:
        transformer (transformer object): The transformer whose output shape is to be logged.
        name (str): An optional name for the transformer, used in logging.

    Examples:
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.decomposition import PCA
        >>> iris = load_iris()
        >>> X = iris.data

        # Using ShapeLogger to log the shape after applying PCA
        >>> pca_logger = ShapeLogger(PCA(n_components=2), name='PCA')
        >>> pca_logger.fit_transform(X)

        # Integrating ShapeLogger into a pipeline
        >>> from sklearn.pipeline import Pipeline
        >>> pipeline = Pipeline([
        ...     ('shape_logger', ShapeLogger(PCA(n_components=2), name='PCA')),
        ... ])
        >>> pipeline.fit_transform(X)

    """

    def __init__(self, transformer, name=""):
        """
        Args:
            transformer (transformer object): The transformer to wrap.
            name (str): An optional name for logging.
        """
        self.transformer = transformer
        self.name = name

    def fit(self, X, y=None):
        """Fit the transformer to the data.

        Args:
            X (array-like): The input data.
            y (array-like, optional): The target values (not used).

        Returns:
            self (ShapeLogger): The fitted transformer.
        """
        self.transformer.fit(X, y)
        self._fitted = True
        return self

    def transform(self, X):
        """Apply the transform method of the transformer and log the output shape.

        Args:
            X (array-like): The data to transform.

        Returns:
            The transformed data, with the output shape logged.
        """
        X_transformed = self.transformer.transform(X)
        logger.info(f"{self.name} output shape: {X_transformed.shape}")
        return X_transformed
