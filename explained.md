# transaction Pipeline

This is a beautifully structured data engineering pipeline. Looking at `pipeline.py` and the newly provided configuration files, the `transactions_pipeline` acts as an expansive feature factory. It takes raw, disparate data sources and systematically builds them up into a single, massive **Master Table** (`ftr_master`).

Here is a detailed, node-by-node breakdown of exactly what is happening as the data flows through this pipeline.

### Phase 1: Data Validation & Ingestion (Raw to Intermediate)

Before any logic is applied, the pipeline ensures the incoming data is pristine.

1. **`schema_validation`**: This node acts as a strict gatekeeper. It reads the YAML configuration (`transactions.schemas`) and compares it against the incoming Parquet and CSV files (Invoices, Customers, Vehicles, Branches, WorldPop). If a column name has changed or a data type is wrong, the pipeline hard-fails immediately, preventing corrupted data from cascading downstream.
2. **`columns_formatting`**: Converts the raw data into the `02_intermediate` layer. It casts columns to their correct PySpark types, enforces uppercase strings, trims whitespace, and converts complex Unix `bigint` timestamps into standard PySpark `DATE` and `TIMESTAMP` formats.

### Phase 2: Building the Foundation (Primary Layer)

This phase establishes the core entities and the timeline over which features will be calculated.

3. **`create_spine`**: This is the heart of the time-series logic.
* It creates a composite `_id` by combining a vehicle's `plate_number` and the customer's `mobile` number.
* It filters out bad data (e.g., fake phone numbers or invalid 7-character license plates).
* It generates a "Spine"—a monthly timeline for every valid customer starting from their very first transaction up to `2026-04-01`. Even if a customer didn't visit in a specific month, a row exists for them.


4. **`create_prm_customers`, `create_prm_vehicles`, `create_prm_branches**`: These nodes deduplicate the intermediate tables. Notably, the customer node creates a `preferred_language` feature, assigning 'AR' (Arabic) to a hardcoded list of Middle Eastern/North African nationalities, and 'EN' (English) to the rest.
5. **`create_prm_geolocation`**: A computationally heavy node that runs in Pandas/Geopandas using `sklearn.neighbors.BallTree`. It takes the GPS coordinates of every Petromin branch and calculates the distance to OpenStreetMap (OSM) points of interest (hospitals, ATMs, cafes) and WorldPop density grids. It creates features counting exactly how many amenities are within a 300m, 500m, and 1000m radius of each branch.
6. **`create_prm_transactions`**: Merges the cleaned transactions onto the monthly spine. It classifies raw product names into major categories like `oil`, `oil_synthetic`, `tyres`, `batteries`, and `brake`.

### Phase 3: Domain-Specific Feature Engineering

This is where the business logic is applied to create predictive features for the machine learning models.

7. **`create_sales_features`**: Aggregates financial and behavioral metrics at a monthly level.
* **Financials:** Total net sales, profit, discounts, and average basket size.
* **Behavioral:** Time of day (morning, afternoon, night) and day of the week (weekend vs. weekday) to understand *when* the customer prefers to visit.


8. **`create_mileage_features`**: The most complex node in the pipeline.
* It calculates **Mileage Per Day (MPD)** by looking at the difference in odometer readings between visits.
* It uses historical MPD to forecast what the customer's mileage is *right now*, even if they haven't visited recently.
* It joins this forecasted mileage against the `Servicing Rules` (the OEM config) to flag if a specific car (e.g., a Toyota Camry) is mathematically "due" or "overdue" for a 5,000km Mineral oil change or a 10,000km Synthetic oil change.


9. **`create_geolocation_features`**: If a customer visits multiple branches, this node identifies their "most visited" branch for the month and maps that specific branch's OSM spatial features (population density, nearby cafes) to the customer's profile.
10. **`create_vehicle_features`**: Calculates the `car_age` (current year minus model year) and groups them into buckets (0-2 years, 3-5 years, 10+ years). It also flags the car's origin (`is_japanese`, `is_chinese`, `is_american`).
11. **`create_special_trx_features`**: Looks at the `raw_promos` table to flag if a transaction was driven by a specific marketing campaign or a Preventive Maintenance Service (PMS).
12. **`create_segment_features`**: Applies classic RFM (Recency, Frequency, Monetary) business logic to bucket customers. It creates tags like `is_new_joiner`, `is_loyal`, `is_lost`, and an interesting `is_promo_hunter` tag for customers who *only* ever visit when there is an active discount.
13. **`create_branches_features`**: Flags if the customer's last visit was in a "PAC" city (a major hub like Riyadh, Jeddah, Dammam, Mecca).

### Phase 4: Target Definition & Dynamic Windows

14. **`create_churn_features`**: Defines the target variable for the downstream machine learning model. A customer is flagged as `is_churn = 1` if they stop visiting **AND** their forecasted mileage indicates they are overdue for an expected oil change. It also creates forward-looking targets (`target_churn_1`, `target_churn_2`) to predict churn 1, 2, and 3 months into the future.
15. **`create_ftr_transactions` (`ftr_windows_sales`)**: Notice that the python file `d_feature_windows_transactions.py` is virtually empty. It delegates the work to a custom internal engine (`feature_generation.v1.nodes...`). It reads the `transactions_pipeline.yml` config and automatically generates rolling averages, sums, and maxes for the past 5, 11, and 23 months for dozens of columns.

### Phase 5: The Master Table Join

16. **`create_ftr_master`**: The final step. It takes the Spine and performs a massive sequence of `LEFT JOIN` operations, merging all the disparate feature tables (Sales, Mileage, Vehicle, Segmentation, Geolocation, Windows, Churn) into one flat, incredibly wide DataFrame.
* *Optimization note:* Because joining a table this massive in Spark can cause memory crashes (spills), the code elegantly splits the customers into `active` and `inactive` groups, performs the joins on the smaller subsets, and then unions them back together at the end.



The output of this pipeline is `ftr_master.parquet`. When this finishes, Kedro passes this flat table directly into your downstream data science pipelines (`predict_churn` and `mileage_model`) for ML training and inference.

Would you like to dive deeper into the code of a specific node, such as the mathematical outlier detection in the mileage calculations, or move on to how the `churn_model` pipeline consumes this master table?


# Predict Churn

The `predict_churn_pipe` is the inference (scoring) phase of your project. While the `transactions_pipeline` generated historical features for everyone, the `predict_churn` pipeline isolates the customers currently at risk and uses a pre-trained machine learning model to score their likelihood of churning in the near future.

Here is a detailed breakdown of how this pipeline executes, what AI model is powering it, and where the training logic lives.

### Step-by-Step Execution of `predict_churn`

This pipeline consists of two distinct nodes defined in `pipeline.py` and implemented in `nodes.py`.

#### 1. Filtering the Target Population (`filter_with_conditions`)

* **Input:** `transactions.ftr_master@spark` (the massive feature table outputted by your data engineering step).
* **What it does:** It filters the master table down to the exact cohort of customers you want to score. It reads the `filter_churn_predict` list from `conf/base/parameters/data_science/churn/model_churn.yml`.
* **The Logic:** Based on the YAML config, it applies date bounds: `_observ_end_dt >= "2026-02-01"` and `< "2026-03-01"`. This means it isolates a single specific month (February 2026) to generate predictions for.
* **Output:** `churn.mdl_predict_filtered@spark` (A smaller dataset of just the current active customers).

#### 2. Generating the Predictions (`predict_churn`)

* **Input:** Kedro automatically transcodes the filtered Spark DataFrame into a Pandas DataFrame (`churn.mdl_predict_filtered@pandas`). It also loads the pre-trained model and the 0.5 probability threshold from your parameters.
* **What it does:** * It extracts the exact list of features the model was trained on using `model.feature_names_in_`.
* It calls `model.predict_proba()` to get the percentage likelihood that the customer will churn.
* It applies the threshold: if the probability is > 0.5, `churn_prediction` becomes 1 (Churn), otherwise 0 (Retained).
* It categorizes the probabilities into 5% buckets (e.g., "0.85 - 0.90") using `pd.cut()` so business stakeholders can easily filter for the "highest risk" deciles.


* **Output:** `churn.mdl_churn_predicted` (A table saved back to disk containing the customer IDs alongside their churn scores).

---

### The AI Model Being Used

The model doing the predicting is a **LightGBM Classifier** (a highly efficient gradient boosting tree framework).

**How do we know?**
If you look at `conf/base/catalog/data_science/churn.yml`, the input key `churn.mdl_estimator` maps directly to this file:
`filepath: data/06_models/churn/model/lgb_classifier.pkl`

Additionally, the tuning parameters defined in `model_churn.yml` (`colsample_bytree`, `lambda_l1`, `learning_rate`, `max_depth`) are classic hyperparameter arguments for LightGBM/XGBoost models.

### Where is the Training Logic?

The `predict_churn` pipeline *only* does inference; it does not train the model.

If you want to investigate how the model was trained and what data it learned from, you need to look at the **`churn_model`** pipeline (which is registered in your `pipeline_registry.py`).

Here is how the training data is structured based on your configuration files:

* **Training Data Source:** It uses the exact same `ftr_master` table, but filters it using different time windows.
* **Training Filter:** In `model_churn.yml`, the `filter_model_conditions` show that the model trains on historical data where `_observ_end_dt >= "2022-01-01"` and `< "2024-01-01"`.
* **Validation Filter:** It validates the model's accuracy on data between `2024-09-01` and `2025-05-01` (`filter_churn_validation`).
* **Target Variable:** The model is trained to predict `target_churn_1` (whether the customer churned exactly 1 month after the observation date).


# Model Training Code

This `churn_model` pipeline is a full-scale machine learning training routine. It takes the output of the data engineering phase, performs feature selection, hyperparameter tuning via cross-validation, and finally trains and evaluates a LightGBM classifier.

Here is a detailed, node-by-node breakdown of the `churn_model` pipeline and exactly what is happening under the hood.

---

### 1. `filter_master_table` & `drop_unused_columns`

* **What it does:** The pipeline starts by taking `transactions.ftr_master@spark` (the giant, flat table created by data engineering). `filter_master_table` applies the SQL-like conditions defined in `model_churn.yml` (e.g., `_observ_end_dt < "2024-01-01"`, `acc_churn_past_2_next_0_months < 2`) to isolate the specific cohort of customers needed for training.
* **Cleaning:** `drop_unused_columns` removes features that are either mathematically leaky (like `target_churn_2`) or simply unnecessary for the model, based on the `cols_to_drop` parameter.

### 2. `generate_cv_indexes`

* **What it does:** This is a crucial step for time-series modeling. It creates the exact indices (row numbers) that will be used to train and test the model during cross-validation.
* **The Logic (`get_panel_data_cross_validation_indexes`):**
* It uses `sklearn.model_selection.TimeSeriesSplit`.
* It splits the data into 5 folds (`cv_splits: 5`).
* Crucially, it enforces a 2-month gap (`months_gap: 2`) between the training data and the test data. This is done to simulate reality: in a real-world scenario, you don't have perfect, up-to-the-minute data when you deploy a model.
* *Note:* The code contains logic for undersampling (`RandomUnderSampler`), but according to `model_churn.yml`, `undersampling_rate` is `null`, so it currently trains on the raw, imbalanced dataset.



### 3. `select_features`

* **What it does:** This node reduces the massive list of potential features (`universe_of_features`) down to the 20 most predictive ones.
* **The Logic (`select_features`):**
* **Phase 1 (mRMR):** It runs an initial pass using **mRMR** (Minimum Redundancy Maximum Relevance). It samples 1% of the data and selects 20 features that are highly correlated with churn (relevance) but have low correlation with each other (redundancy).
* **Phase 2 (Sequential Feature Selection):** It instantiates a baseline LightGBM model. It then runs `sklearn.feature_selection.SequentialFeatureSelector` in a `forward` direction. It iteratively adds features to the model, evaluating them against a custom **Kolmogorov-Smirnov (KS)** scorer.
* It outputs the final list of 20 optimal features as `churn.mdl_selected_features`.
* *Observation:* In your current configuration, `params:selected_features` is already hardcoded with 20 features. This node might be currently bypassed or used only for exploratory runs, as the downstream `train_model` node explicitly requests `params:selected_features` rather than the output of this node.



### 4. `train_model`

* **What it does:** This node performs the actual cross-validation and trains the final LightGBM model.
* **The Logic (`train_model` & `cv_function`):**
* It loops through the 5 time-series splits generated earlier.
* For each split, it trains a LightGBM model (`lgb.LGBMClassifier`) and tests it on the holdout month.
* It calculates a suite of metrics for every fold: Precision, Recall, ROC AUC, PR AUC, and KS Statistic.
* *Note on Hyperparameter Tuning:* The pipeline definition shows that the `tune_hyperparameters` node is currently commented out. Instead, `train_model` uses a hardcoded dictionary of `model_hyperparameters` (`learning_rate: 0.1`, `n_estimators: 100`, `colsample_bytree: 0.7`).
* Finally, it trains one last model on the *entire* training dataset.
* It also generates a `shap.TreeExplainer` so the business can understand why the model makes specific predictions.



### 5. `filter_validation_churn` & `validation_node`

* **What it does:** This tests the finalized model on a completely unseen, out-of-time dataset.
* **The Logic (`validation_node` & `validate_model`):**
* It filters the master table using `filter_churn_validation` (data from `2024-09-01` to `2025-05-01`).
* It runs the trained model over these new months and calculates the same suite of metrics (Precision, Recall, ROC AUC, KS) to see how the model degrades over time.
* It generates SHAP visualizations (Beeswarm plots, bar charts) to show global feature importance.



---

### How to Retrain the Model

If you want to retrain the model with fresh data, the process is heavily reliant on updating `conf/base/parameters/data_science/churn/model_churn.yml`.

**Step 1: Update the Date Filters**
Open `model_churn.yml` and slide your training and validation windows forward.

```yaml
filter_model_conditions:
  - _observ_end_dt < "2025-01-01" # Move this forward
  - _observ_end_dt >= "2023-01-01" # Move this forward
  - target_churn_1 IS NOT NULL
  - below_estimated_mineral_oil_change > 0
  - acc_churn_past_2_next_0_months < 2
  - is_active_sum_past_11_next_0_months > 0

filter_churn_validation:
  - _observ_end_dt < "2026-03-01" # Validate on the most recent data
  - _observ_end_dt >= "2025-09-01" 

```

**Step 2: (Optional) Re-enable Hyperparameter Tuning**
Currently, the pipeline is using hardcoded hyperparameters inside `train_model`. If you want it to actually search for better parameters, you need to:

1. Open `src/petromin/pipelines/data_science/churn_model/pipeline.py`.
2. Uncomment the `tune_hyperparameters` node.
3. Change the `model_hyperparameters` input in the `train_model` node to accept the output of the tuner: `"model_hyperparameters": "churn.mdl_tuning_params"`.

**Step 3: (Optional) Run Feature Selection**
Currently, `train_model` uses the hardcoded list `params:selected_features`. If you want the model to discover new features automatically:

1. Change the `columns` input in the `train_model` node from `"params:selected_features"` to the output of the selection node: `"churn.mdl_selected_features"`.

**Step 4: Execute the Pipeline**
Run the following command in your terminal:

```bash
kedro run --pipeline=churn_model

```

This will generate a new `lgb_classifier.pkl` file in your `data/06_models/churn/model/` directory, which will be used the next time you run `predict_churn`.


# Model training Guide

Retraining this model is actually quite straightforward because the project is built using Kedro and highly parameterized via YAML files. You won't even need to touch the Python code unless you want to completely overhaul the machine learning algorithm.

Based on the directory structure and the `model_churn.yml` config you provided, here is your step-by-step guide to retraining the LightGBM churn model with fresh data.

### Step 1: Update the Training Time Windows

In churn prediction, your model is highly sensitive to time. You need to tell the model which historical months to look at for training, and which later months to use for validation.

Open this file:
`conf/base/parameters/data_science/churn/model_churn.yml`

Look for the **`filter_model_conditions`** block. Currently, it is configured to train on older data:

```yaml
filter_model_conditions:
  - _observ_end_dt < "2024-01-01" # To Be update before training
  - _observ_end_dt >= "2022-01-01"
  - target_churn_1 IS NOT NULL
  - below_estimated_mineral_oil_change > 0
  - acc_churn_past_2_next_0_months < 2
  - is_active_sum_past_11_next_0_months > 0

```

Since you are currently in **March 2026**, you'll want to slide this window forward to capture more recent consumer behavior. For example, you might change it to train on 2024 through the end of 2025:

```yaml
  - _observ_end_dt >= "2024-01-01"
  - _observ_end_dt < "2026-01-01" 

```

You should also slide the **`filter_churn_validation`** dates forward to validate on early 2026 data.

### Step 2: (Optional) Adjust the Hyperparameter Search

In that exact same `model_churn.yml` file, scroll down to the `tuning_params_distribution` section.

```yaml
tuning_iterations: 150
tuning_metric_to_maximize: recall
tuning_params_distribution:
  colsample_bytree:
  - 0.7
  - 0.9
  learning_rate:
  - 0.01
  - 0.05
  - 0.1
  - 0.2
  - 0.5
  max_depth:
  - 5
  - 3
  - -1
  n_estimators:
  - 300
  - 100

```

When you trigger the training pipeline, it will automatically run 150 iterations testing different combinations of these exact parameters to find the model that yields the highest **Recall**. If you want the training to go faster, drop `tuning_iterations` to 50. If you want it to explore more options, add values to these lists.

### Step 3: Run the Training Pipeline

Once your YAML configuration is saved, open your terminal at the root of the project (`./churn-pipeline/`).

Since your `pipeline_registry.py` registers the training pipeline as `"churn_model": churn_pipe`, you simply run:

```bash
kedro run --pipeline=churn_model

```

**What happens when you run this?**

1. It grabs the massive `ftr_master` table you built in the data engineering phase.
2. It runs the scripts inside `src/petromin/pipelines/data_science/churn_model/nodes.py`.
3. It filters the rows based on your new date conditions.
4. It selects the exact features listed under `features:` in your YAML.
5. It runs the hyperparameter tuning over your predefined grid.
6. It saves the newly trained, optimized model back to disk as a pickle file located at: `data/06_models/churn/model/lgb_classifier.pkl` (as defined in `catalog/data_science/churn.yml`).

### Step 4: Run Inference with the New Model

Once the training finishes successfully, the new `.pkl` artifact overwrites the old one. Now, when you run your original command:

```bash
kedro run --pipeline=full_mileage_model

```

...the `predict_churn` pipeline will automatically load your newly minted model to score today's active customers.

---

# Mileage Pipe

The `mileage_pipe` is the final, highly actionable stage of your `full_mileage_model`. While the first pipeline built the historical features and the second pipeline predicted churn probability, this third pipeline synthesizes all of that into a clean, business-ready dataset.

It takes the raw mathematical forecasts and turns them into marketing segments, A/B testing groups, and warranty flags.

Here is a detailed, node-by-node breakdown of exactly what is happening in the `mileage_pipe`.

### 1. Cohort Isolation (`filter_mileage_forecast`)

* **Input:** `transactions.ftr_master@spark` (the giant data engineering table).
* **What it does:** It filters the massive master table down to the exact month you want to generate a campaign or report for.
* **The Logic:** It looks at `conf/base/parameters/data_science/mileage/model_mileage.yml` and applies the `filter_mileage_forecast` conditions. In your config, this isolates the data between `2026-03-01` and `2026-04-01` (specifically grabbing March 2026).

### 2. A/B Test Setup & Customer Rollups (`prepare_mileage_forecast`)

Since one customer (one phone number) might own multiple cars (multiple license plates), this PySpark node rolls data up from the *vehicle level* to the *customer level*.

* **A/B Testing (`campaign_group`):** It extracts the phone number (`mobile`) from the `_id` string and looks at the last two digits. If the digits are 00, 01, 02, 03, or 04, the customer is put in a `"control"` group; otherwise, they go to `"test"`. This is a classic, deterministic way to do randomized control trials for marketing campaigns.
* **Customer-Level Aggregation:** It groups by `mobile` and sums up the revenue, visits, and promo usage across *all* cars owned by that person.
* **Primary Car Flagging:** It joins this aggregated data back to the vehicle data and creates an `is_highest_revenue_car` flag so the business knows which car is the primary driver of revenue for a multi-car owner.
* **Due Strings:** It checks 10 different product categories (Mineral Oil, AC Services, Spark Plugs, etc.) and concatenates their individual due flags into one easy-to-read string (e.g., `__mineral_oil__air_filter`).

### 3. Business Logic & Churn Integration (`forecast_mileage`)

Kedro hands the data over to Pandas for this final, heavily operational node. This node translates raw numbers into discrete business categories and integrates the churn predictions.

* **Memory Optimization:** It starts by downcasting the boolean/flag columns to `Int8` (a nullable 8-bit integer in Pandas) to dramatically save RAM during processing.
* **Business Segmentation:** It assigns clean, string-based labels to various features using `np.select` and `pd.cut`:
* **Loyalty:** Categorizes customers into `New Joiner`, `Uncommited`, `Potential Loyal`, `Loyal`, `Lost`, or `Gone`.
* **Price Sensitivity:** Categorizes them into `Full price`, `Promo hunter`, or `Mixed price`.
* **Bucketing:** It groups continuous variables into human-readable buckets (e.g., Car Age goes into "0-2", "3-5", "10+", Mileage Per Day goes into "25-50", "50-75", etc.).
* **Deciles:** It splits revenue into 10 quantiles so you can easily target your top 10% spenders.


* **Integrating the Churn Model:** It takes the `churn_probability` and `churn_bucket` outputted by the `predict_churn` pipeline and merges it onto this table. **This is the core value of the pipeline:** you now have a table showing exactly who is due for an oil change *and* exactly how likely they are to churn if you don't intervene.
* **Warranty Logic (`is_on_warranty_period`):** It applies hardcoded Original Equipment Manufacturer (OEM) business rules. For example, Cadillacs and Lexuses under 5 years old get flagged as being in-warranty. Hyundais and Kias under 6 years get the flag, Cherys under 7 years, etc.
* **Final Cleanup:** It drops duplicated columns, ensuring it keeps the version of the column with the least null values, and returns a lean, focused DataFrame.

### The Final Output

The output of this pipeline is `mileage.mileage_forecast` (saved as a Parquet file).

This file is essentially a "hit list" for a CRM, marketing team, or operational dashboard. Every row is a vehicle, complete with its forecasted mileage, its exact A/B test group, what specific services it is due for, its warranty status, its owner's preferred language, and its probability of churning.

---

# Insight on Cust. Categorization

While the final text labels ("Loyalty", "Promo hunter", etc.) are applied at the very end of the pipeline inside `mileage_model/nodes.py` using Pandas, the **actual mathematical brain** behind these segments lives upstream in the PySpark data engineering layer.

Specifically, this logic is calculated in `transactions_pipeline/nodes/d_feature_customer_transactions.py` inside the `create_segment_features` function.

Petromin isn't just bucketing people based on flat transaction counts; they are using a highly dynamic, personalized approach that looks at **how much a specific customer drives**.

Here is the exact, in-depth breakdown of how the pipeline builds these categories.

### Part 1: The Foundation (Personalized Expected Visits)

Before categorizing loyalty, the pipeline calculates exactly how many times a customer *should* be visiting a year, based on their personal driving habits.

* It looks at `customer_avg_mileage_per_day` and multiplies it by 365 to get their annual mileage.
* **Expected Mineral Oil Visits:** Annual mileage divided by 5,000 km.
* **Expected Synthetic Oil Visits:** Annual mileage divided by 10,000 km.

This means a taxi driver who drives 200 km a day has a totally different loyalty threshold than someone who only drives 20 km a day to the office.

---

### Part 2: Loyalty Segmentation Logic

The pipeline evaluates these conditions sequentially to assign a 1 or 0 flag to every customer. In the final stage, `np.select` picks the first flag that evaluates to true.

**1. New Joiner (`is_new_joiner`)**

* **The Math:** `total_number_of_visits == 1` AND `months_since_last_visit < 12` AND `months_since_first_visit < 12`
* **The Meaning:** This is a customer who has only ever visited Petromin exactly once in their entire history, and that single visit happened within the last year.

**2. Loyal (`is_loyal`)**

* **The Math:** Must have visited at least 3 times in the last 12 months (`total_number_of_visits_last_12_months >= 3`) **AND** they must meet or exceed their personalized expected visit count for the type of oil they use (e.g., if their daily mileage dictates they should get 4 synthetic oil changes a year, they must have visited at least 4 times).
* **The Meaning:** The ultimate healthy customer. They come frequently, and they come *as often as their car mathematically needs service*.

**3. Potential Loyal (`is_potential_loyal`)**

* **The Math:** Must have visited at least 2 times in the last 12 months **AND** they meet at least **70%** of their expected visits for their oil type (e.g., `visits > expected_visits * 0.7`) **AND** they are not already flagged as `is_loyal`.
* **The Meaning:** These customers are good, but they are "leaking" some oil changes to competitors. They come to Petromin for most of their needs, but their mileage implies they are getting serviced somewhere else occasionally.

**4. Uncommited (`is_uncommited`)**

* **The Math:** Visited at least 1 time in the last 12 months **AND** `months_since_last_visit < 12` **AND** they don't qualify for New Joiner, Loyal, or Potential Loyal.
* **The Meaning:** The tricky middle ground. They are an active customer (seen in the last year, have more than 1 lifetime visit), but they visit far less frequently than their odometer says they should.

**5. Lost (`is_lost`)**

* **The Math:** `months_since_last_visit > 11` AND `<= 24`.
* **The Meaning:** They haven't been seen in 1 to 2 years. They are effectively churned, but recent enough that a win-back campaign might still work.

**6. Gone (`is_gone`)**

* **The Math:** `months_since_last_visit > 24`.
* **The Meaning:** It has been over 2 years since their last transaction. They have likely sold the car, moved away, or are permanently loyal to a competitor.

---

### Part 3: Price Sensitivity Logic

To figure out what kind of marketing offer will work on a customer, the pipeline looks at the past 12 months of transactions and compares their `total_number_of_visits` against their `total_number_of_promo_visits` (transactions explicitly flagged as using a campaign/discount package).

**1. Promo Hunter (`is_promo_hunter`)**

* **The Math:** `total_visits > 0` AND `total_visits == promo_visits`
* **The Meaning:** Extremely price-sensitive. 100% of their visits in the last year involved a discount. Sending them an ad without a coupon code is likely a waste of money.

**2. Full Price (`is_full_price`)**

* **The Math:** `total_visits > 0` AND `promo_visits == 0`
* **The Meaning:** Completely price-insensitive. They pay list price every time. Sending these customers discount codes actually hurts business margins because they would have paid full price anyway.

**3. Mixed Price (`is_mixed_price`)**

* **The Math:** Everything else (not a Promo Hunter, not a Full Price customer).
* **The Meaning:** They use coupons when available but are willing to pay full price when necessary.

### How it comes together

In the final Pandas `forecast_mileage` node, it literally just takes these 1/0 flags and turns them into text. So if a row has `is_potential_loyal = 1` and `is_promo_hunter = 1`, that customer hits the CRM as a **"Potential Loyal - Promo Hunter"**, making it incredibly easy for the marketing team to know exactly how to target them!

