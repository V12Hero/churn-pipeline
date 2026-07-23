# CLAUDE.md — Petromin

Scope note: this repo is large, but **only two things are actually run**:

1. `python ingestion_pipeline.py` — pulls raw data from SQL Server into `data/01_raw/`
2. `kedro run --pipeline full_mileage_model` — builds features, predicts churn, produces the mileage forecast

Everything documented below is limited to those two paths. Other pipelines registered in
`src/petromin/pipeline_registry.py` (segmentation, footprint, nptb, osm, worldpop, mastertable,
churn *training*, …) are **not in scope** — do not touch or "fix" them unless explicitly asked.

---

## 1. The monthly run

This is a monthly batch. Both steps are run manually, in order, after bumping ~4 date knobs.

Everything runs inside the **`BTL` conda env** (`/home/ammar/miniconda3/envs/BTL`, Python 3.12) —
`conda activate BTL` first, always. Both steps are run by hand in a tmux session, not by an agent:
they take hours, hold a 60 GB Spark driver, and two concurrent runs would collide on the same
catalog paths. Hand over the command rather than executing it.

### Step 1 — ingestion

```bash
python ingestion_pipeline.py
```

Writes to `data/01_raw/`:

| Output | Written by |
| --- | --- |
| `raw_branches.parquet` | `ingestion_general()` |
| `raw_customers.parquet` | `ingestion_general()` |
| `raw_vehicles.parquet` | `ingestion_general()` |
| `raw_invoices.parquet` | `ingestion_invoices()` |
| `raw_invoices_items_PE.parquet` (+ `_PE_files/` monthly shards) | `ingestion_invoice_items_PE()` |
| `raw_invoices_items_PAC.parquet` (+ `_PAC_files/` monthly shards) | `ingestion_invoice_items_PAC()` |
| `raw_invoices_items.parquet` | `ingestion_items()` |
| `transactions_origin.parquet` | `prepare_transactions()` ← **this is what Kedro reads** |

Runtime is roughly 1.5h end-to-end (per the timing comments in `main()`).

### Step 2 — the model

```bash
stdbuf -oL -eL /usr/bin/time -v kedro run --pipeline full_mileage_model 2>&1 | tee -a logs/ammar-run.log
```

`stdbuf -oL -eL` keeps Spark's progress output line-buffered through the pipe; `/usr/bin/time -v`
appends peak RSS and wall clock at the end; the log is **appended** to `logs/ammar-run.log`, so that
file holds several runs — read from the bottom.

Final output: `data/06_models/mileage/mileage_forecast.parquet`.

To resume after a mid-pipeline failure, add `--from-nodes "<node name from the WARNING line>"`.
Kedro prints the right node to use in the failure message.

---

## 2. Date knobs — bump these before every run

Four values must be moved together. Getting the relative offsets wrong is the most common failure
mode, and it fails **silently** (empty filter → empty forecast).

| # | File | Key | Example (Aug-2026 run) |
| --- | --- | --- | --- |
| 1 | `ingestion_pipeline.py:807` | `min_date` in `main()` | `"2026-06-01"` |
| 2 | `conf/base/parameters/data_engineering/transactions_pipeline.yml:509` | `transactions.spine_end_dt` | `2026-09-01` |
| 3 | `conf/base/parameters/data_science/churn/model_churn.yml:186` | `filter_churn_predict` | `>= "2026-07-01"`, `< "2026-08-01"` |
| 4 | `conf/base/parameters/data_science/mileage/model_mileage.yml` | `filter_mileage_forecast` | `>= "2026-08-01"`, `< "2026-09-01"` |

Rules that tie them together:

- **`spine_end_dt` is exclusive-ish.** `create_spine` does
  `pd.date_range(start_dt, end_dt, freq='1M')` then snaps to `last_day`. So `spine_end_dt: 2026-09-01`
  produces a last snapshot of `_observ_end_dt = 2026-08-31`. Set it to the **first day of the month
  after** the month you want forecast.
- **The churn window is one month BEHIND the mileage window.** This is not a bug.
  `forecast_mileage` (`mileage_model/nodes.py:486`) shifts the churn frame forward:
  `churn_df["_observ_end_dt"] + 1 day + MonthEnd()`, so churn's `2026-07-31` becomes `2026-08-31`
  and joins onto the mileage row for August. If both filters use the same month, the churn merge
  produces all-null `churn_probability` / `churn_bucket`.
- **`min_date` is an incremental-load trigger, not a modelling date.** It only limits the
  *invoice items* extraction (the expensive part); dimension tables and invoice headers are always
  pulled from `MIN(ModifiedOn)`, i.e. full history. Keep it a couple of months behind the target
  month so late-modified invoices are re-pulled. It does **not** need to line up with the filters above.
- `filter_*` conditions are raw Spark SQL strings evaluated with `f.expr()` against `_observ_end_dt`.

`explained.md` has a longer prose walkthrough of the ingestion script and the transactions pipeline.

---

## 3. `ingestion_pipeline.py`

Standalone ETL script — **not a Kedro pipeline**, despite the name. Bridges the live SQL Server
(`MACDB`) to `data/01_raw/`.

- Credentials: `SERVER` / `DATABASE` / `USERNAME` / `PASSWORD` from `.env` (via `python-dotenv`),
  connected with `pyodbc` + `ODBC Driver 18 for SQL Server`. Never hardcode or echo these.
- Everything is pulled **month-by-month** (`WHERE YEAR(ModifiedOn)=… AND MONTH(ModifiedOn)=…`) to
  avoid loading the whole table into RAM. Invoice items go one step further and write each month to
  its own parquet shard on disk before being read back as one frame.
- Every table exists twice — `v_X` (brand `PE`) and `v_PAC_X` (brand `PAC`). Each ingest function
  pulls both, tags `StationBrand`, and concatenates.
- **`ingest_promos()` is commented out of `ingestion_general()`** (`ingestion_pipeline.py:730-732`),
  so `data/01_raw/raw_promos.parquet` is *stale* — it is not refreshed by a run, but the Kedro
  pipeline still reads it (`create_special_trx_features`). Expected; don't "fix" without asking.
- Most functions coerce every column to `str` before writing. Casting back to numeric happens
  downstream. Don't be surprised by object dtypes in `01_raw`.
- `ingest_vehicles()` is a huge hand-maintained normalisation table for `Make` / `Model` typos
  (cashier free-text), plus the `vehicle_brand_level` price tiering. New misspellings get appended
  to those `.str.replace` chains.

---

## 4. `full_mileage_model` — composition

Defined at `src/petromin/pipeline_registry.py:110`:

```python
"full_mileage_model": transactions_pipeline + predict_churn_pipe + mileage_pipe
```

⚠️ **Import shadowing:** `pipeline_registry.py:11` imports `transactions_pipeline` as
`transactions_pipe`, then **line 14 rebinds the same name** to `new_transactions_pipeline`.
Line 14 wins. The pipeline that actually runs is
`src/petromin/pipelines/data_engineering/new_transactions_pipeline/`.
The older `transactions_pipeline/` package is dead code for this path.

### 4a. `new_transactions_pipeline` (namespace `transactions`)

`.../new_transactions_pipeline/pipeline.py` — 15 nodes, all namespaced `transactions.`.
Node functions live in `nodes/`, named by layer (`a_raw` → `b_intermediate` → `c_primary_*` →
`d_feature_*` → `e_master_table`).

Flow:

```
raw_transactions, raw_customers, raw_vehicles, raw_branches, raw_world_pop
  └─ schema_validation        (validate_schema, params:schemas)
  └─ columns_formatting       (reformat_columns) → int_transactions, int_customers/vehicles/branches, prm_world_pop
       ├─ create_spine        → prm_spine        (the _id × _observ_end_dt grid; everything joins to this)
       ├─ create_prm_customers / _vehicles / _branches
       ├─ create_prm_geolocation  (+ geospatial POI/highways/subways)
       └─ create_prm_transactions (drops branches_to_drop / products_to_drop)
            ├─ create_sales_features        → ftr_sales
            ├─ create_vehicle_features      → ftr_vehicle
            ├─ create_mileage_features      → ftr_mileage    (needs ftr_vehicle + raw_servicing_rules)
            ├─ create_geolocation_features  → ftr_geolocation
            ├─ create_special_trx_features  → ftr_special_trxs  (reads raw_promos)
            ├─ create_churn_features        → ftr_churn
            ├─ create_segment_features      → ftr_segment
            ├─ create_branches_features     → ftr_branches
            └─ create_ftr_transactions      → ftr_windows_sales
                 └─ create_ftr_master (ftr_join_dfs_spine) → transactions.ftr_master@spark
```

The two keys used everywhere: **`_id`** (customer–vehicle unit of analysis, formatted
`something__mobile__…`) and **`_observ_end_dt`** (month-end snapshot date).

`create_mileage_features` (`nodes/d_feature_mileage_transactions.py`, the biggest node at ~845 lines)
is where `customer_mileage_forecast`, `customer_mileage_last_forecast`, `expected_number_of_visits_*`
and the whole `is_due_<product>` family come from, driven by `data/01_raw/Servicing Rules.csv`.

### 4b. `predict_churn` (namespace `churn`)

`src/petromin/pipelines/data_science/predict_churn/` — **inference only, no training.**

| Node | In | Out |
| --- | --- | --- |
| `filter_predict_churn` | `transactions.ftr_master@spark`, `params:filter_churn_predict` | `churn.mdl_predict_filtered@spark` |
| `predict_churn` | `churn.mdl_predict_filtered@pandas`, `params:features`, `churn.mdl_estimator`, `params:probability_threshold` | `churn.mdl_churn_predicted` |

- `churn.mdl_estimator` is a **pre-trained pickle** at `data/06_models/churn/model/lgb_classifier.pkl`.
  It is not produced by this pipeline — if it's missing, run the separate `churn_model` pipeline
  (out of scope here).
- The node uses `model.feature_names_in_`, **not** `params:features`, to select columns. The
  `selected_cols` argument is effectively ignored. If `ftr_master` loses a column the model was
  trained on, this raises a `KeyError` — that's the signal.
- `probability_threshold: 0.5` (`model_churn.yml:195`). Also emits a 20-bucket `churn_bucket`.

### 4c. `mileage_model` (namespace `mileage`)

`src/petromin/pipelines/data_science/mileage_model/`.

| Node | In | Out |
| --- | --- | --- |
| `filter_mileage_forecast` | `transactions.ftr_master@spark`, `params:filter_mileage_forecast` | `mileage.forecast_master_filtered` |
| `prepare_mileage_forecast` | above | `mileage.forecast_base@spark` |
| `forecast_mileage` | `mileage.forecast_base@pandas`, `churn.mdl_churn_predicted`, `params:mileage.closed_station_list` | `mileage.mileage_forecast` |

- `prepare_mileage_forecast` (Spark): derives `mobile` from `split(_id, "__")[1]`, assigns
  `campaign_group` = `control` if the mobile's last two digits are `00`–`04` else `test`, aggregates
  per-mobile totals (a customer can own several vehicles), flags `is_highest_revenue_car`,
  computes `crossing_10k_threshold` from the 10 000 km buckets, and concatenates the per-product
  `is_due_*` flags into one `is_due` string.
- `forecast_mileage` (pandas): loyalty/price segments, revenue deciles, all the buckets
  (`car_age_bucket`, `mpd_bucket`, …), `is_station_closed`, the churn merge, warranty-period and
  `is_pms_final_flag` logic. Then filters `columns_to_keep`, de-duplicates repeated column names
  (`columns_to_keep` lists `is_due_mineral_oil` etc. twice — `drop_dupes_keep_most_data` handles it),
  and **keeps only the single latest `_observ_end_dt`**.
- It returns `[filtered_forecast_df]` (a list) because the node declares `outputs=[...]`.
- The whole function is memory-tuned: `.to_numpy()` + `np.select` instead of chained `np.where`,
  `Int8`/categorical downcasts, explicit `del` + `gc.collect()`. The original readable version is
  kept commented out above it (`nodes.py:160-334`). Preserve that style when editing — the frame
  is tens of millions of rows.
- `mileage.closed_station_list` lives in `transactions_pipeline.yml:809` (currently `[]`), **not**
  in `model_mileage.yml`.

---

## 5. Config layout

| What | Where |
| --- | --- |
| Transactions catalog (raw → ftr_master) | `conf/base/catalog/transactions_pipeline.yml` |
| Churn catalog | `conf/base/catalog/data_science/churn.yml` |
| Mileage catalog | `conf/base/catalog/data_science/mileage.yml` |
| Transactions params (schemas, spine dates, drop-lists, `closed_station_list`) | `conf/base/parameters/data_engineering/transactions_pipeline.yml` |
| Churn params (features, filter, threshold) | `conf/base/parameters/data_science/churn/model_churn.yml` |
| Mileage params (forecast filter only) | `conf/base/parameters/data_science/mileage/model_mileage.yml` |
| Spark tuning | `conf/base/spark.yml` |
| Credentials | `conf/local/credentials.yml`, `.env` — never commit or print |

Kedro 0.19.12, `OmegaConfigLoader`, Python `>=3.9,<3.12`. Spark session is set up by
`SparkHooks` in `src/petromin/hooks.py` (registered in `settings.py`).

### `@spark` / `@pandas` transcoding

Several datasets exist twice pointing at the same path, e.g. `transactions.ftr_master@spark` and
`@pandas`, or `churn.mdl_predict_filtered@spark`/`@pandas`. Kedro writes with one and reads with
the other. If you add a node that consumes a Spark output as pandas, **add the matching `@pandas`
catalog entry** rather than converting inside the node.

`conf/base/spark.yml` is tuned for a single big local box (`driver.memory: 60g`,
`maxResultSize: 40g`, 600 shuffle partitions, scratch in `.tmp/`). Spark work stays lazy; the
`.count()` calls inside `filter_with_conditions` are deliberate logging checkpoints and are
expensive.

---

## 6. Known issue — branch coordinates are NULL (open as of 2026-07-22)

`create_prm_geolocation` fails with:

```
ValueError: Found array with 0 sample(s) (shape=(0, 2)) while a minimum of 1 is required.
```

**Cause.** `v_Branch` / `v_PAC_Branch` in MACDB now return NULL for `Latitude` and `Longitude` on
all 1119 branches. `ingest_branches()` str-casts every column, so `raw_branches.parquet` holds the
literal string `"None"`; the `branches` schema casts to FLOAT → all NaN → the
`dropna(axis=0, how="any")` at `c_primary_geolocation.py:112` empties the frame → the BallTree at
line 127 gets a `(0, 2)` array.

This is **upstream, not a code regression** — the Aug-2025 snapshot `data/01_raw/branches.parquet`
has real float64 coordinates. The fix belongs with whoever owns MACDB.

**Do not "fix" it by deleting the geolocation node.** Its outputs feed the churn model
(`avg_n_places_within_*`, `avg_pop_density` in `model_churn.yml:18-25`), and `predict_churn` selects
columns via `model.feature_names_in_`, so removing them just moves the failure to a `KeyError`.

**Workaround in use** — reuse the last good geolocation and skip the node:

```bash
kedro run --pipeline full_mileage_model --from-nodes "transactions.create_prm_transactions"
```

`data/03_primary/prm_geolocation` (Jun 24 2026, 170,872 rows, 1104 branch_ids) covers 1104 of
today's 1119 branches. The 15 newer ones get null geo features, which slightly degrades their churn
scores. This is a one-month patch and the cache stales further every month.

Related trap: that `dropna(how="any")` has *always* silently discarded branches with missing
coordinates. It only became a crash when the proportion hit 100%, so partial degradation in earlier
months went unnoticed.

## 7. Gotchas

- **Empty output, no error.** Almost always a date-filter mismatch (§2). Check the
  `filtered master shape: (N, M)` log line from `filter_with_conditions` — if `N == 0`, the window
  is wrong or the spine doesn't reach that month.
- **All-null `churn_probability`.** The churn and mileage filters are on the same month; churn must
  be one month earlier (§2).
- Both `filter_with_conditions` implementations (`predict_churn/nodes.py:16` and
  `mileage_model/nodes.py:20`) are duplicated near-identical copies. Fix bugs in both.
- There are stray `# breakpoint()` lines left in the DS nodes. Harmless; leave them unless cleaning up.
- `data/`, `logs/`, `info.log`, `venv/`, `__pycache__/` are working artefacts — never commit them.
- The `ingestion_pipeline.py` SQL is f-string-interpolated. Table names are hardcoded literals, so
  it's fine as-is, but don't extend it to take user input without parameterising.
