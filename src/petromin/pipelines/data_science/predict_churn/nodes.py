"""
This is a boilerplate pipeline 'predict_churn'
generated using Kedro 0.18.12
"""
import pyspark
import pyspark.sql.functions as f
from typing import List, Union, Dict
import numpy as np
import pandas as pd

import logging

logger = logging.getLogger(__name__)


def filter_with_conditions(
        df:pyspark.sql.DataFrame,
        conditions: List[str]
)->pyspark.sql.DataFrame:
    filter_cond= f.lit(True)
    if conditions:
        for cond in conditions:
            msg = f"adding condition {cond}"
            logger.info(msg)
            filter_cond = filter_cond & (f.expr(cond))

    out = df.filter(filter_cond)

    logger.info(
        f"filtered master shape:  ({out.count()}, {len(out.columns)})",
    )


    return out.orderBy(["_id","_observ_end_dt"])


def predict_churn(
        df: pd.DataFrame,
        selected_cols: List,
        model,
        threshold: float
):
    out = df.copy()
    # breakpoint()
    features_names = model.feature_names_in_
    out['churn_probability'] = model.predict_proba(df[features_names])[:,-1]
    out['churn_prediction'] = np.where(
        out['churn_probability'] > threshold, 
        1,
        0
    )

    churn_buckets_labels = [
        "0.00 - 0.05", "0.05 - 0.10", "0.10 - 0.15", "0.15 - 0.20", "0.20 - 0.25", "0.25 - 0.30", "0.30 - 0.35", "0.35 - 0.40", "0.40 - 0.45", "0.45 - 0.50",
        "0.50 - 0.55", "0.55 - 0.60", "0.60 - 0.65", "0.65 - 0.70", "0.70 - 0.75", "0.75 - 0.80", "0.80 - 0.85", "0.85 - 0.90", "0.90 - 0.95", "0.95 - 1.00",
        "1.00 - 1.05"
    ]

    out['churn_bucket'] = pd.cut(out["churn_probability"], bins=np.arange(0, 1.1, 0.05), labels=churn_buckets_labels, right=False)

    return out
