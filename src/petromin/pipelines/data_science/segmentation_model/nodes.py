"""Model Segmentation nodes."""
import logging
import typing as tp

import numpy as np
import pandas as pd
import pyspark
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from petromin.modelling.models.segmentation import build_simple_segmentation_model
from petromin.modelling.reproducibility.set_seed import seed_file

from pyspark import SparkConf
from pyspark.sql import SparkSession, functions as f, Window, DataFrame

from functools import reduce, partial

logger = logging.getLogger(__name__)

spark = SparkSession.builder.getOrCreate()


def segmentation_model_fit(df: pyspark.sql.DataFrame, params: tp.Dict[str, tp.Any]) -> Pipeline:
    """
    Fits a segmentation model to the provided DataFrame using specified parameters.

    This function first converts a PySpark DataFrame to a Pandas DataFrame, fills missing values,
    and then constructs and fits a clustering pipeline. The pipeline is built using dimensionality
    reduction models, scalers, and a clustering model, all of which are loaded from the provided
    parameters.

    Args:
        df (pyspark.sql.DataFrame): The DataFrame on which the segmentation model is to be fitted.
        params (Dict[str, Any]): A dictionary containing parameters for constructing the pipeline.
        This includes feature column names, dimensionality reduction model parameters,
        scaler parameters, and the clustering model.

    Returns:
        Pipeline: A fitted sklearn pipeline that includes preprocessing steps and a clustering model.

    Example:
        ```python
        from pyspark.sql import SparkSession

        # Initialize a Spark session
        spark = SparkSession.builder.appName("SegmentationModel").getOrCreate()

        # Example DataFrame
        df = spark.createDataFrame([
            (1, "value1", 3.0),
            (2, "value2", 4.5),
            # Add more rows as needed
        ], ["id", "category", "feature"])

        # Example parameters
        params = {
            "features": {
                "transactional_columns": ["feature"],
                "geospatial_columns": ["category"]
            },
            "dimensionality_reduction": {
                "transactional_columns": {"PCA": {"n_components": 2}},
                "geospatial_columns": {"TruncatedSVD": {"n_components": 2}}
            },
            # Add other necessary parameters like scalers and clustering model
        }

        # Fit the model
        pipeline = segmentation_model_fit(df, params)
        ```

    Note:
        - The function assumes that missing values in the DataFrame can be filled with 0.
        This assumption should be verified for the specific use case.
        - It is essential that the `params` dictionary is correctly formatted and
        contains all necessary sub-dictionaries and parameters.
    """
    seed_file()

    features = [item for sublist in params["features"].values() for item in sublist]
    missing_columns = set(features).difference(set(df.columns))
    if len(missing_columns) > 0:
        raise ValueError(f"Missing columns in master table dataset: {missing_columns}")

    logger.info("start pipeline setting")
    pipeline = build_simple_segmentation_model(params)
    logger.info("start pipeline training")
    pipeline = pipeline.fit(df)
    logger.info("check pipeline training")
    check_is_fitted(pipeline)

    try:
        model = pipeline.steps[-1][1]["cluster"]
    except Exception:
        model = pipeline.steps[-1][1]

    logger.info("Plot results and get centroids")
    fig = model.get_inertia_plot()
    centroids = _get_proxy_centroid_stores(df, model)
    fig.show()

    wcss_feature_importance = model.wcss_feature_importances_.reset_index()
    unsupervised_feature_importance = model.unsupervised_feature_importance_.reset_index()

    # breakpoint()

    return dict(
        pipeline=pipeline,
        params=params,
        fig=fig,
        centroids=centroids,
        wcss_feature_importance=wcss_feature_importance,
        unsupervised_feature_importance=unsupervised_feature_importance,
    )


def _get_proxy_centroid_stores(df: pd.DataFrame, model: Pipeline) -> tp.Dict[str, str]:
    """
    Get proxy centroid stores for each cluster from a clustering model.

    This function calculates and returns a dictionary mapping cluster labels (keys)
    to the corresponding proxy centroid stores (values) based on the provided DataFrame
    and clustering model.

    Args:
        df (pd.DataFrame): The DataFrame containing data points used for clustering.
        model (Pipeline): The Scikit-learn Pipeline model representing the clustering algorithm.

    Returns:
        dict: A dictionary mapping cluster labels to the proxy centroid store IDs.

    Example:
        >>> from sklearn.pipeline import Pipeline
        >>> import pandas as pd
        >>> data = pd.DataFrame({"_id": ["StoreA", "StoreB", "StoreC"], "feature1": [1.0, 2.0, 3.0]})
        >>> clustering_model = Pipeline([("cluster", KMeans(n_clusters=3))])
        >>> centroids = _get_proxy_centroid_stores(data, clustering_model)
        >>> print(centroids)
        {'0': 'StoreA', '1': 'StoreB', '2': 'StoreC'}

    Note:
        The function assumes that the DataFrame (`df`) contains a column named "_id" that
        uniquely identifies each store, and the `model` is a clustering model that has
        already been fitted to the data.

    """
    centroids = {}
    for key, value in model.centroid_indexes.items():
        store = df["_id"].iloc[value]
        centroids[key] = store
        logger.info(f"Centroid {key} has {store} as centroid ")
    return centroids


def segmentation_model_inference(df: pd.DataFrame, params: tp.Dict[str, tp.Any], pipeline: Pipeline) -> pd.DataFrame:
    """
    Applies a fitted segmentation pipeline to a DataFrame and appends the cluster
    assignments as a new column.

    This function predicts cluster assignments for each row in the provided DataFrame using the provided
    fitted pipeline. The cluster assignments are added as a new column to the DataFrame.
    The name of this new column is dynamically generated based on the class name of the last
    step in the pipeline (typically the clustering model). Additionally, the function logs the
    normalized value counts of the cluster assignments.

    Parameters:
        df (pd.DataFrame): The DataFrame to which the segmentation model will be applied.
        pipeline (Pipeline): A fitted sklearn pipeline that includes preprocessing steps
            and a clustering model.

    Returns:
        pd.DataFrame: The input DataFrame augmented with a new column containing cluster
        assignments.

    Example:
        ```python
        import pandas as pd
        from sklearn.cluster import KMeans
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        # Sample data
        data = {'feature1': [1, 2, 3, 4, 5], 'feature2': [5, 4, 3, 2, 1]}
        df = pd.DataFrame(data)

        # Sample pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=2)),
            ('kmeans', KMeans(n_clusters=2))
        ])
        pipeline.fit(df)

        # Apply the model and get the DataFrame with cluster assignments
        result_df = segmentation_model_inference(df, pipeline)
        print(result_df)
        ```

    Note:
        - The function assumes that the pipeline has already been fitted. An error will
        occur if the pipeline is not fitted.
        - The function logs the normalized value counts of the cluster assignments,
        which can be useful for analyzing the distribution of the clusters.
    """
    # add cluster column
    # breakpoint()
    df["cluster_id"] = pipeline.predict(df)

    logger.info("Clustering distribution normalized:")
    logger.info(df["cluster_id"].value_counts(normalize=True))
    logger.info("Clustering distribution:")
    logger.info(df["cluster_id"].value_counts(normalize=False))

    # assign cluster id
    # df["cluster_id"] = "cluster_id_" + df["cluster_id"].apply(str)

    unique_cluster_ids = df["cluster_id"].unique()

    for id in unique_cluster_ids:
        df[f"cluster_id_{id}"] = np.where(df["cluster_id"] == id, 1, 0)

    return df

def segmentation_model_inference_spark(df: pyspark.sql.DataFrame, params: tp.Dict[str, tp.Any], pipeline: Pipeline) -> pyspark.sql.DataFrame:

    # Broadcast the model
    broadcasted_model = spark.sparkContext.broadcast(pipeline)


    @f.udf('integer')
    def predict_udf(*cols):
        return int(broadcasted_model.value.predict((cols,)))

    list_of_columns = df.columns
    df_prediction = df.withColumn('cluster_id', predict_udf(*list_of_columns))

    unique_cluster_ids = df_prediction.select("cluster_id").distinct().rdd.flatMap(list).collect()

    for id in unique_cluster_ids:
        df_prediction = df_prediction.withColumn(
            f"cluster_id_{id}",
            f.when(
                f.col("cluster_id") == id,
                1
            ).otherwise(0)
        )

    return df_prediction

def ftr_join_dfs_segmentations(
    *args, **kwargs,
) -> pyspark.sql.DataFrame:

    df_list = list(kwargs.values())

    union_by_name = partial(DataFrame.unionByName, allowMissingColumns=True)

    joined_df = reduce(union_by_name, df_list)

    microsegment_df = joined_df.drop_duplicates().withColumn(
        "loyalty_segment",
        f.when(
            f.col("is_loyal") == 1,
            "loyal"
        ).when(
            f.col("is_potential_loyal") == 1,
            "potential_loyal"
        ).when(
            f.col("is_uncommited") == 1,
            "uncommited"
        ).when(
            f.col("is_new_joiner") == 1,
            "new_joiner"
        ).when(
            f.col("is_lost") == 1,
            "lost"
        ).when(
            f.col("is_gone") == 1,
            "gone"
        ).otherwise("uncommited")
    ).withColumn(
        "price_segment",
        f.when(
            f.col("is_promo_hunter") == 1,
            "promo_hunter"
        ).when(
            f.col("is_full_price") == 1,
            "full_price"
        ).when(
            f.col("is_mixed_price") == 1,
            "mixed_price"
        ).otherwise("mixed_price")
    ).withColumn(
        "macrosegment",
        f.concat_ws("__", f.col("loyalty_segment"), f.col("cluster_id"))
    ).withColumn(
        "microsegment",
        f.concat_ws("__", f.col("loyalty_segment"), f.col("price_segment"), f.col("cluster_id"))
    )

    onehot_macrosegment_df = microsegment_df.groupBy(
        "_id", "_observ_end_dt"
    ).pivot(
        "macrosegment"
    ).agg(
        f.count("cluster_id")
    ).fillna(0)

    onehot_microsegment_df = microsegment_df.groupBy(
        "_id", "_observ_end_dt"
    ).pivot(
        "microsegment"
    ).agg(
        f.count("cluster_id")
    ).fillna(0)

    out = microsegment_df.join(
        onehot_macrosegment_df,
        on=["_id", "_observ_end_dt"],
        how="left"
    ).join(
        onehot_microsegment_df,
        on=["_id", "_observ_end_dt"],
        how="left"
    )

    return out.orderBy(["_id", "_observ_end_dt"])