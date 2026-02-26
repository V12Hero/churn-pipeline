import typing as tp

import logging

import pyspark.sql
from pyspark.sql import functions as f

from functools import reduce

logger = logging.getLogger(__name__)


def ftr_join_dfs_spine(
    spine: pyspark.sql.DataFrame,
    **kwargs
):


    columns_list = []
    for df_name, df in kwargs.items():
        logger.info(f"{df_name[:11]} -\t\t\t shape:\t\t\t({df.count()}, {len(df.columns)})")
        print("columns overlap:")
        print(set(columns_list).intersection(set(list(df.columns))))
        columns_list += df.columns

    ftr_segment = kwargs["ftr_segment"]

    filter_active_clients = (f.col("is_lost") < 1) & (f.col("is_gone") < 1)
    active_clients_df = ftr_segment.filter(filter_active_clients).select("_id", "_observ_end_dt")

    active_spine = spine.select(
        "_id", "_observ_end_dt"
    ).distinct().join(
        active_clients_df,
        on=["_id", "_observ_end_dt"],
        how="inner"
    )

    logger.info(f"Active Spine -\t\t\t shape:\t\t\t({active_spine.count()}, {len(active_spine.columns)})")

    list_dfs =[active_spine] + list(kwargs.values())
    active_ftr_master = reduce(lambda x, y: x.join(y, ["_id", "_observ_end_dt"], how="left"), list_dfs)

    filter_inactive_clients = (f.col("is_lost") > 0) | (f.col("is_gone") > 0)
    inactive_clients_df = ftr_segment.filter(filter_inactive_clients).select("_id", "_observ_end_dt")

    inactive_spine = spine.select(
        "_id", "_observ_end_dt"
    ).distinct().join(
        inactive_clients_df,
        on=["_id", "_observ_end_dt"],
        how="inner"
    )

    logger.info(f"Inactive Spine -\t\t\t shape:\t\t\t({inactive_spine.count()}, {len(inactive_spine.columns)})")

    list_dfs =[inactive_spine] + list(kwargs.values())
    inactive_ftr_master = reduce(lambda x, y: x.join(y, ["_id", "_observ_end_dt"], how="left"), list_dfs)

    out = active_ftr_master.union(inactive_ftr_master)

    return out.orderBy(["_id", "_observ_end_dt"])


# # nodes/e_master_table.py
# import typing as tp
# import logging
# import time

# import pyspark.sql
# from pyspark.sql import functions as f
# from pyspark.storagelevel import StorageLevel
# from functools import reduce

# logger = logging.getLogger(__name__)

# JOIN_KEYS = ["_id", "_observ_end_dt"]


# def _select_keys_and_nondupe(df: pyspark.sql.DataFrame, kept_cols: tp.Set[str]) -> pyspark.sql.DataFrame:
#     """
#     Keep join keys + only columns not already selected (avoid collisions/shuffles on drop/rename later).
#     Output columns remain identical after the full left-join cascade; we just avoid carrying duplicates repeatedly.
#     """
#     new_cols = [c for c in df.columns if c not in kept_cols or c in JOIN_KEYS]
#     return df.select(*[f.col(c) for c in new_cols])


# def _prep(df: pyspark.sql.DataFrame, num_parts: int) -> pyspark.sql.DataFrame:
#     """
#     Align partitioning on the join keys to reduce shuffle during the cascade of left joins.
#     Persist to disk (not memory) to save heap pressure. No logic/row change.
#     """
#     return (
#         df.repartitionByRange(num_parts, *[f.col(k) for k in JOIN_KEYS])
#           .persist(StorageLevel.DISK_ONLY)
#     )


# def ftr_join_dfs_spine(
#     spine: pyspark.sql.DataFrame,
#     **kwargs
# ):
#     t0 = time.perf_counter()
#     logger.info("▶ Starting feature join build (no upfront actions).")

#     spark = spine.sql_ctx.sparkSession

#     # ---- Session safety knobs (don’t change logic; improve shuffle balance) ----
#     # Adaptive execution & skew join mitigation help large shuffles a lot.
#     spark.conf.set("spark.sql.adaptive.enabled", "true")
#     spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
#     # Make tasks smaller to reduce per-task spill size:
#     default_shuffle_parts = int(spark.conf.get("spark.sql.shuffle.partitions", "200"))
#     if default_shuffle_parts < 800:
#         spark.conf.set("spark.sql.shuffle.partitions", "1200")
#     # Optional: smaller file chunks can reduce "No space left" bursts on constrained disks.
#     # spark.conf.set("spark.sql.files.maxPartitionBytes", str(64 * 1024 * 1024))  # 64MB

#     # If the app didn't set a checkpoint dir, set a local one (harmless if already set).
#     sc = spark.sparkContext
#     try:
#         # Accessing conf throws if unset; we’ll set a default in that case.
#         _ = sc.getCheckpointDir()
#     except Exception:
#         sc.setCheckpointDir("/tmp/spark-checkpoints")

#     num_parts = int(spark.conf.get("spark.sql.shuffle.partitions"))
#     logger.info(f"• Using shuffle partitions: {num_parts}")

#     # ---- Prep inputs: keys-only spine, aligned partitioning ----
#     # (distinct keys keeps the same semantics; you joined on keys anyway)
#     spine_keys = spine.select(*JOIN_KEYS).distinct()
#     spine_prep = _prep(spine_keys, num_parts)
#     logger.info(f"• Spine prepared | partitions={spine_prep.rdd.getNumPartitions()} | cols={spine_prep.columns}")

#     # ---- Trim feature columns to avoid dup projections; align partitions; persist to DISK ----
#     kept_cols: tp.Set[str] = set(JOIN_KEYS)
#     features = {}
#     for name, df in kwargs.items():
#         df_trim = _select_keys_and_nondupe(df, kept_cols)
#         kept_cols.update(df_trim.columns)
#         features[name] = _prep(df_trim, num_parts)
#         logger.info(f"• Feature '{name}' prepped | parts={features[name].rdd.getNumPartitions()} | cols={len(df_trim.columns)}")

#     # We need 'ftr_segment' to derive the keys that exist (active OR inactive). Logic unchanged.
#     ftr_segment = features["ftr_segment"]

#     # ---- SINGLE pass over keys: active ∪ inactive ≡ all keys present in ftr_segment ----
#     # Your previous union(active, inactive) == distinct keys from ftr_segment.
#     keys_all = _prep(ftr_segment.select(*JOIN_KEYS).distinct(), num_parts)
#     logger.info(f"• Keys from ftr_segment prepared | parts={keys_all.rdd.getNumPartitions()}")

#     # Inner-join spine with the complete key set (same rows as doing active/inactive separately then union).
#     filtered_spine = (
#         spine_prep.join(keys_all, on=JOIN_KEYS, how="inner")
#                   .persist(StorageLevel.DISK_ONLY)
#     )
#     logger.info(f"• Filtered spine ready | parts={filtered_spine.rdd.getNumPartitions()} | cols={len(filtered_spine.columns)}")

#     # (Optional) Broadcast hints for truly small inputs (≤ autoBroadcastJoinThreshold)
#     # Example:
#     # for small_name in ("prm_geolocation", "ftr_branches"):
#     #     if small_name in features:
#     #         features[small_name] = features[small_name].hint("broadcast")

#     # ---- Cascade left-joins once (instead of twice) ----
#     feature_list = [v for _, v in features.items()]  # preserve all features
#     def join_all(base_df, dfs):
#         return reduce(lambda x, y: x.join(y, JOIN_KEYS, how="left"), [base_df] + dfs)

#     t_join = time.perf_counter()
#     master_once = join_all(filtered_spine, feature_list).persist(StorageLevel.DISK_ONLY)
#     logger.info(f"• Joined all features in one pass | parts={master_once.rdd.getNumPartitions()} | elapsed={time.perf_counter()-t_join:.1f}s")

#     # Cut lineage to keep the final sort/write lighter (rows/schema unchanged)
#     master_ckpt = master_once.checkpoint(eager=True)
#     logger.info("• Checkpointed master dataframe to truncate lineage.")

#     # ---- Final ordering (kept for exact parity) ----
#     # We pre-range-partitioned on JOIN_KEYS, so the global order shuffle is better balanced.
#     t_sort = time.perf_counter()
#     out = master_ckpt.orderBy(JOIN_KEYS)

#     # If you do NOT require a single global order, uncomment the following two lines
#     # and comment out the global orderBy above. Output rows are identical; order only differs
#     # across partitions. This change dramatically reduces spill risk.
#     # out = master_ckpt.repartitionByRange(num_parts, *[f.col(k) for k in JOIN_KEYS]) \
#     #                  .sortWithinPartitions(*JOIN_KEYS)

#     logger.info(f"• Final ordering done in {time.perf_counter()-t_sort:.1f}s")
#     logger.info(f"✓ Completed feature master build in {time.perf_counter()-t0:.1f}s")
#     return out
