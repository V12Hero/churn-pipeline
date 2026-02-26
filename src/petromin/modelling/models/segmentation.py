import logging
import typing as tp
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import euclidean
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import DBSCAN, OPTICS, KMeans, MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import euclidean_distances, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import tqdm

from petromin.modelling.transformers.columns_transformer import ColumnSelector
from petromin.modelling.transformers.logger import ShapeLogger
from petromin.python_utils.load.object_inyection import load_estimator, load_object

logger = logging.getLogger(__name__)


def build_segmentation_model(
    params: tp.Dict[str, tp.Union[tp.Dict[str, tp.Any], BaseEstimator, TransformerMixin]]
) -> Pipeline:
    """Build a segmentation model pipeline with K-Means clustering.

    This function constructs a segmentation model pipeline with K-Means clustering based on the specified parameters.
    The resulting pipeline includes a clustering algorithm for segmenting data based on the specified parameters.

    Args:
        params (dict): A dictionary containing parameters for building the segmentation model pipeline.

    Returns:
        sklearn.pipeline.Pipeline: A segmentation model pipeline with K-Means clustering.

    Examples:
        Usage example of the build_segmentation_model function with K-Means clustering:

        ```python
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.datasets import make_blobs
        from my_module import load_object

        # Define parameters for the segmentation model with K-Means clustering
        segmentation_params = {
            "features": {
                "feature_set_1": ["feature_1", "feature_2"],
                "feature_set_2": ["feature_3", "feature_4"],
            },
            "scaler": {
                "feature_set_1": {"class": "sklearn.decomposition.PCA", "kwargs": },
                "feature_set_2": {"class": "sklearn.decomposition.PCA", "kwargs": },
            },
            "cluster": {"class": "sklearn.cluster.KMeans", "kwargs": },  # Use K-Means clustering with 3 clusters
        }

        # Build the segmentation model pipeline with K-Means clustering
        segmentation_pipeline = build_segmentation_model(segmentation_params)

        # Generate synthetic data for demonstration
        X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

        # Fit and transform data using the pipeline
        segmented_labels = segmentation_pipeline.fit_predict(X)
        ```
    """
    features = {key: params["features"][key] for key in params["features"].keys()}

    use_embedders = params.get("embedders", None)
    if use_embedders is not None:
        embedders = {
            key: load_object(params["embedders"][key]) for key in params["embedders"].keys()
        }
        scalers = {key: load_object(params["scaler"][key]) for key in params["scaler"].keys()}
        preprocessing_pipelines = [
            (
                f"embedding_process_{key}_data",
                Pipeline(
                    [
                        ("selector", ColumnSelector(features[key])),
                        ("scaler", scalers[key]),
                        (
                            key,
                            ShapeLogger(
                                embedders[key],
                                name=f"{embedders[key].__class__.__name__}",
                            ),
                        ),
                    ],
                ),
                features[key],
            )
            for key in features.keys()
        ]

    if use_embedders is None:
        preprocessing_pipelines = [
            (
                f"column_selector_process_{key}_data",
                Pipeline(
                    [
                        ("selector", ColumnSelector(features[key])),
                    ],
                ),
                features[key],
            )
            for key in features.keys()
        ]

    if params["cluster"]["scale_last_step"]:
        cluster_pipe = Pipeline(
            [
                ("seg_scaler", load_object(params["cluster"]["scaler"])),
                ("cluster", load_estimator(params["cluster"]["model"])),
            ]
        )
    else:
        cluster_pipe = load_estimator(params["cluster"]["model"])

    pipeline = Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=preprocessing_pipelines,
                    transformer_weights={
                        f"preprocessing_process_{key}_data": 1 for key in features.keys()
                    },
                    n_jobs=-1,
                ),
            ),
            ("cluster_model", cluster_pipe),
        ]
    )
    return pipeline


def build_simple_segmentation_model(
    params: tp.Dict[str, tp.Union[tp.Dict[str, tp.Any], BaseEstimator, TransformerMixin]]
) -> Pipeline:
    features = list(set([name for sublist in params["features"].values() for name in sublist]))

    if params["cluster"]["scale_last_step"]:
        pipeline = Pipeline(
            [
                ("selector", ColumnSelector(features)),
                ("seg_scaler", load_object(params["cluster"]["scaler"])),
                ("cluster", load_estimator(params["cluster"]["model"])),
            ]
        )
    else:
        pipeline = load_estimator(params["cluster"]["model"])

    return pipeline


class ElbowClusterSelector(BaseEstimator, TransformerMixin, ClusterMixin):
    """ElbowClusterSelector.

    ElbowClusterSelector selects the optimal number of clusters for K-Means clustering
    using the elbow method.

    Args:
        max_clusters : int, optional (default=10)
            The maximum number of clusters to consider when finding the optimal number
            of clusters using the elbow method.

    Attributes:
        optimal_num_clusters : int
            The optimal number of clusters selected using the elbow method.

        fitted : bool
            Indicates whether the model has been fitted.

    Examples:
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

    >>> # Create and set up the ElbowClusterSelector instance
    >>> cluster_selector = ElbowClusterSelector(max_clusters=10)

    >>> # Fit the ElbowClusterSelector
    >>> cluster_selector.fit(X)

    >>> # Create a pipeline with StandardScaler and the cluster selector
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> pipeline = Pipeline([
    ...     ('scaler', StandardScaler()),  # Example pre-processing step
    ...     ('cluster_selector', cluster_selector),
    ... ])

    >>> # Fit the entire pipeline
    >>> pipeline.fit(X)

    >>> # Now you can use the entire pipeline for predictions, e.g., pipeline.predict(X)
    """

    def __init__(self, min_clusters=2, max_clusters=10):
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.is_fitted = False
        self.model_args = {"init": "k-means++", "random_state": 42, "tol": 1e7}

    def fit(self, X, y=None):
        wcss = []  # Within-cluster sum of squares
        graph = []

        logger.info(f"Shape of input: {X.shape}")

        # # db = DBSCAN(eps=0.3, min_samples=10000, n_jobs=-1).fit(X)
        # db = OPTICS(n_jobs=-1).fit(X)
        # labels = db.labels_

        # # Number of clusters in labels, ignoring noise if present.
        # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        # n_noise_ = list(labels).count(-1)

        # print("Estimated number of clusters: %d" % n_clusters_)
        # print("Estimated number of noise points: %d" % n_noise_)

        # initialize with 1 cluster
        X_sample = X.sample(frac=0.01)

        for i in tqdm.tqdm(range(self.max_clusters, self.min_clusters-1, -1)):
            # logger.info(f"Performing clusters with {i} clusters")
            model = KMeans(n_clusters=i, **self.model_args)
            model.fit(X)

            cluster_labels = model.predict(X_sample)
            silhouette_avg = silhouette_score(X_sample, cluster_labels, n_jobs=-1)
            # silhouette_avg_list.append(silhouette_avg)

            # print(np.mean(silhouette_avg_list), np.std(silhouette_avg_list))
            # silhouette_avg = np.mean(silhouette_avg_list)
            wcss.append(model.inertia_)
            graph.append([i, model.inertia_, silhouette_avg])

            logger.info(f"Kmeans inertia - n cluster {i}: {model.inertia_} - {silhouette_avg}")

            # for batch_size in [5000, 10000, 50000, 100000]:

            #     mbk = MiniBatchKMeans(
            #         init="k-means++",
            #         n_clusters=i,
            #         batch_size=batch_size,
            #         n_init=50,
            #         max_no_improvement=100,
            #         verbose=0,
            #     )
            #     mbk.fit(X)
            #     cluster_labels = mbk.predict(X_sample)
            #     silhouette_avg = silhouette_score(X_sample, cluster_labels, n_jobs=-1)
            #     logger.info(f"Mini Batch Kmeans inertia - {batch_size}: {mbk.inertia_}  - {silhouette_avg}")

        inertia = np.sum(euclidean_distances(X.to_numpy(), X.to_numpy().mean(axis=0).reshape(1, -1), squared=True))
        graph.append([1, inertia, 0])

        logger.info(f"Kmeans inertia - n cluster 1: {inertia} - 0")

        self.graph = graph

        # Use the elbow method to find the optimal number of clusters
        self.optimal_num_clusters = self.find_optimal_num_clusters()
        logger.info(f"Optimal number of clusters: {self.optimal_num_clusters}")
        # Fit the K-Means model with the optimal number of clusters
        self.model = KMeans(n_clusters=self.optimal_num_clusters, **self.model_args)
        self.model = self.model.fit(X)
        predicted = self.model.predict(X)
        predicted_df = X.copy()
        predicted_df["cluster"] = predicted
        predicted_df_agg = predicted_df.groupby("cluster").mean()
        predicted_df_mean = predicted_df.mean().to_frame().T
        predicted_df_mean.name = "avg_pop"
        print(pd.concat([predicted_df_agg, predicted_df_mean], axis=0).to_string())
        self.is_fitted = True
        self.labels_ = self.model.labels_
        self.inertia_ = self.model.inertia_
        self.train_data = X
        self._get_cluster_centroids_index()
        self.get_feature_imp_wcss_min()
        self.get_feature_imp_unsup2sup()

        return self

    def predict(self, X):
        # Assign cluster labels to the input data
        cluster_labels = self.model.predict(X)
        return cluster_labels

    def _get_cluster_centroids_index(self):
        # Get the cluster centroids
        self.centroid_indexes = {}
        if isinstance(self.train_data, pd.DataFrame):
            iterable_data = self.train_data.values
        else:
            iterable_data = self.train_data

        for centroid in self.model.cluster_centers_:
            min_centroid_distance = np.inf

            for i, data_point in enumerate(iterable_data):
                distance = euclidean(centroid, data_point)
                if distance < min_centroid_distance:
                    centroid_index = i
                    min_centroid_distance = distance
            label = self.labels_[centroid_index]
            self.centroid_indexes[f"cluster_id_{label}"] = centroid_index
        logger.info(f"Centroids dictionary -> {self.centroid_indexes}")
        return self

    def get_feature_imp_wcss_min(self):
        labels = self.model.n_clusters
        centroids = self.model.cluster_centers_

        if isinstance(self.train_data, pd.DataFrame):
            self.ordered_feature_names = self.train_data.columns
        else:
            self.ordered_feature_names = pd.DataFrame(self.train_data).columns

        centroids = np.vectorize(lambda x: np.abs(x))(centroids)
        sorted_centroid_features_idx = centroids.argsort(axis=1)[:, ::-1]

        dfs = []
        for label, centroid in zip(range(labels), sorted_centroid_features_idx):
            ordered_cluster_feature_weights = centroids[label][sorted_centroid_features_idx[label]]
            ordered_cluster_features = [self.ordered_feature_names[feature] for feature in centroid]

            df_importance = pd.DataFrame(
                [ordered_cluster_features, ordered_cluster_feature_weights]
            ).T
            df_importance.columns = ["feature", f"cluster_id_{label}"]
            df_importance[f"cluster_id_{label}"] = (
                df_importance[f"cluster_id_{label}"]
                / df_importance[f"cluster_id_{label}"].sum()
                * 100
            )
            dfs.append(df_importance.set_index("feature"))

        cluster_feature_weights = pd.concat(dfs, axis=1)
        self.wcss_feature_importances_ = cluster_feature_weights
        self.wcss_feature_importances_["mean_importance"] = self.wcss_feature_importances_[
            list(self.wcss_feature_importances_.columns)
        ].sum(axis=1)
        self.wcss_feature_importances_["mean_importance"] = (
            self.wcss_feature_importances_["mean_importance"]
            / self.wcss_feature_importances_["mean_importance"].sum()
            * 100
        )
        self.wcss_feature_importances_ = self.wcss_feature_importances_.sort_values(
            by="mean_importance", ascending=False
        )

        logger.debug("WCSS feature importances: ")
        logger.debug(self.wcss_feature_importances_.head(20))

        for cluster_id in self.wcss_feature_importances_:
            data = self.wcss_feature_importances_[[cluster_id]]
            data = data.sort_values(by=cluster_id, ascending=False)
            top_predictors = list(data.head(10).index)
            logger.info(
                f"Method: WCSS importance | Top 10 predictors for cluster {cluster_id}: {top_predictors}"
            )
            logger.info("\n\n")

        return self.wcss_feature_importances_

    def get_feature_imp_unsup2sup(self):

        if isinstance(self.train_data, pd.DataFrame):
            self.ordered_feature_names = self.train_data.columns
        else:
            self.ordered_feature_names = pd.DataFrame(self.train_data).columns

        X = self.train_data

        X_sample = X.sample(frac=0.01)

        dfs = []
        for label in range(self.model.n_clusters):
            binary_enc = np.vectorize(lambda x: 1 if x == label else 0)(self.labels_[X_sample.index])
            # overfitt classifier to get feature importance
            clf = RandomForestClassifier(n_estimators=500, random_state=42, max_depth=24)
            clf.fit(X_sample, binary_enc)

            sorted_feature_weight_idxes = np.argsort(clf.feature_importances_)[::-1]
            ordered_cluster_features = np.take_along_axis(
                np.array(self.ordered_feature_names), sorted_feature_weight_idxes, axis=0
            )
            ordered_cluster_feature_weights = np.take_along_axis(
                np.array(clf.feature_importances_), sorted_feature_weight_idxes, axis=0
            )

            df_importance = pd.DataFrame(
                [ordered_cluster_features, ordered_cluster_feature_weights]
            ).T
            df_importance.columns = ["feature", f"cluster_id_{label}"]
            df_importance[f"cluster_id_{label}"] = (
                df_importance[f"cluster_id_{label}"]
                / df_importance[f"cluster_id_{label}"].sum()
                * 100
            )
            dfs.append(df_importance.set_index("feature"))

        cluster_feature_weights = pd.concat(dfs, axis=1)

        self.unsupervised_feature_importance_ = cluster_feature_weights
        self.unsupervised_feature_importance_[
            "mean_importance"
        ] = self.unsupervised_feature_importance_[
            list(self.unsupervised_feature_importance_.columns)
        ].sum(
            axis=1
        )
        self.unsupervised_feature_importance_["mean_importance"] = (
            self.unsupervised_feature_importance_["mean_importance"]
            / self.unsupervised_feature_importance_["mean_importance"].sum()
            * 100
        )
        self.unsupervised_feature_importance_ = self.unsupervised_feature_importance_.sort_values(
            by="mean_importance", ascending=False
        )

        logger.debug("Unsupervised method feature importances: ")
        logger.debug(self.unsupervised_feature_importance_.head(10))

        for cluster_id in self.unsupervised_feature_importance_:
            data = self.unsupervised_feature_importance_[[cluster_id]]
            data = data.sort_values(by=cluster_id, ascending=False)
            top_predictors = list(data.head(10).index)
            logger.info(
                f"Method: Unsupervised importance | Top 10 predictors for cluster {cluster_id}: {top_predictors}"
            )
            logger.info("\n\n")

        return self.unsupervised_feature_importance_

    def find_optimal_num_clusters(
        self,
    ):
        graph = pd.DataFrame(self.graph, columns=["number_of_clusters", "inertia", "score"])
        graph = graph.sort_values("number_of_clusters")
        # Calculate the first derivative of the change in wcss
        graph["change_in_wcss"] = graph["inertia"].diff()
        graph["second_derivative"] = graph["change_in_wcss"].diff()
        graph["n_cluster_rank"] = graph["number_of_clusters"].rank(method="dense", ascending=True) + 0.1
        graph["wcss_rank"] = graph["change_in_wcss"].rank(method="dense", ascending=False) + 0.2

        # if np.all(np.isnan(graph["wcss_rank"].values)):
        #     graph["wcss_rank"] = graph["inertia"].rank(method="dense", ascending=True) + 0.2

        graph["score_rank"] = graph["score"].rank(method="dense", ascending=False) + 0.3
        graph["metric_rank"] = (graph["wcss_rank"] + graph["score_rank"])/2

        graph["rank"] = graph["n_cluster_rank"] + graph["metric_rank"]
        # breakpoint()
        # Find the index where the second derivative is maximum
        optimal_num_clusters_index = graph["rank"].idxmin()

        if np.isnan(optimal_num_clusters_index):
            optimal_num_clusters_index = 0

        # The optimal number of clusters can be obtained from the 'number_of_clusters' column
        optimal_num_clusters = graph.loc[optimal_num_clusters_index, "number_of_clusters"]
        return optimal_num_clusters

    def get_inertia_plot(
        self,
    ) -> px.line:
        """Get inertia plot."""
        if self.is_fitted:
            title = f"k-Means Inertia Plot | Optimal number of cluster is {self.optimal_num_clusters} segments"
            graph = pd.DataFrame(self.graph, columns=["number_of_clusters", "inertia", "score"])

            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Add traces
            fig.add_trace(
                go.Scatter(
                    x=graph["number_of_clusters"],
                    y=graph["inertia"],
                    name="inertia data"),
                secondary_y=False,
            )

            fig.add_trace(
                go.Scatter(
                    x=graph["number_of_clusters"],
                    y=graph["score"],
                    name="score data"),
                secondary_y=True,
            )

            # Add figure title
            fig.update_layout(
                title_text=title
            )

            # fig = px.line(
            #     graph,
            #     x="number_of_clusters",
            #     y=["inertia", "score"],
            #     title=title,
            # )

            fig.add_vline(
                x=self.optimal_num_clusters,
                line_dash="dash",
                line_color="black",
                annotation_text="Optimal number of clusters",
            )
            return fig
        else:
            raise ValueError(
                "The model has not been fitted yet. You should fit before getting inertia plot."
            )
