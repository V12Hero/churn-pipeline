"""
This is a boilerplate pipeline 'model_churn'
generated using Kedro 0.18.9
"""
import pprint
from matplotlib import pyplot as plt
import pyspark
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
from typing import Any, List, Union, Dict
import lightgbm as lgb
import pandas as pd
import numpy as np
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import RandomOverSampler, SMOTE
from scipy.stats import ks_2samp
from sklearn.metrics import make_scorer, \
                        precision_score, \
                        recall_score, \
                        precision_recall_curve, \
                        auc, \
                        roc_auc_score, \
                        confusion_matrix
from sklearn.feature_selection import SequentialFeatureSelector
# from funnelai.node import feature_selection as fs
from sklearn.preprocessing import StandardScaler
import pyspark.sql.functions as f

from mrmr import mrmr_classif, spark as mrmr_sp
import shap

import logging

import tqdm

logger = logging.getLogger(__name__)

def filter_with_conditions(
        df:pyspark.sql.DataFrame,
        conditions: List[str]
)->pyspark.sql.DataFrame:
    """
    Filters a PySpark DataFrame based on a list of conditions and orders the results.

    Args:
        df (pyspark.sql.DataFrame): The input DataFrame to be filtered.
        conditions (List[str]): A list of conditions in SQL expression format that will be applied as filters.

    Returns:
        pyspark.sql.DataFrame: The filtered and ordered DataFrame.
    """

    logger.info(
        f"initial master shape:  ({df.count()}, {len(df.columns)})",
    )

    min_date = df.select(f.min("_observ_end_dt")).collect()[0][0]
    max_date = df.select(f.max("_observ_end_dt")).collect()[0][0]

    msg = f"dates: ({min_date}) - ({max_date})"
    logger.info(msg)

    filter_cond= f.lit(True)
    if conditions:
        for cond in conditions:
            filter_cond = filter_cond & (f.expr(cond))
            msg = f"New Condition:  ({cond})"
            logger.info(msg)

            msg = f"after condition shape:  ({df.filter(filter_cond).count()}, )"
            logger.info(msg)

    # breakpoint()

    out = df.filter(filter_cond)

    logger.info(
        f"filtered master shape:  ({out.count()}, {len(out.columns)})",
    )

    out.groupBy("_observ_end_dt").count().orderBy(f.desc("_observ_end_dt")).show(18)

    min_date = out.select(f.min("_observ_end_dt")).collect()[0][0]
    max_date = out.select(f.max("_observ_end_dt")).collect()[0][0]

    msg = f"dates: ({min_date}) - ({max_date})"
    logger.info(msg)

    return out.orderBy(["_id","_observ_end_dt"])

def drop_columns(
        df: pyspark.sql.DataFrame,
        cols_to_drop: List,
        special_cols: List,
):
    """
    Drops specified columns from a PySpark DataFrame, excluding any special columns that should be retained.

    Args:
        df (pyspark.sql.DataFrame): The input DataFrame from which columns will be dropped.
        cols_to_drop (List[str]): A list of column names to be dropped.
        special_cols (List[str]): A list of column names that should not be dropped, even if they are in `cols_to_drop`.

    Returns:
        pyspark.sql.DataFrame: The DataFrame with the specified columns dropped, excluding the special columns.
    """
    final_cols_to_drop=list(set(cols_to_drop)-set(special_cols))

    msg = f"cols to drop: {", ".join(sorted(final_cols_to_drop))}"
    logger.info(msg)

    df = df.drop(*final_cols_to_drop)

    logger.info(
        f"master after dropped cols:  ({df.count()}, {len(df.columns)})",
    )

    return df

def scale_data(
        df: pd.DataFrame,
        target: str,
        special_cols: List,
):
    """
    Scales the data in a Pandas DataFrame, excluding special columns and target column, using a standard scaler.

    Args:
        df (pd.DataFrame): The input DataFrame to be scaled.
        target (str): The target column name that should not be scaled.
        special_cols (List[str]): A list of column names that should not be scaled.

    Returns:
        Tuple[StandardScaler, pd.DataFrame]: A tuple containing the fitted scaler and the scaled DataFrame.
    """
    for i in df.columns:
        if i.endswith("_flag"):
            special_cols.append(i)
    special_cols.append(target)
    scaler=StandardScaler()
    X_special=df[special_cols].copy()
    X = df[list(set(df.columns)-set(special_cols))].copy()
    scaler.fit(X)
    X = pd.DataFrame(scaler.transform(X), columns = X.columns)
    X = pd.concat([X_special, X], axis=1)
    return scaler, X

def get_panel_data_cross_validation_indexes(
        data: pd.DataFrame,
        splits: int,
        date_column: str,
        months_gap: int,
        months_test: int, 
        max_train_size: int  = None,
        undersampling_rate: int = None,
        target_column: str=None,
        ):
    """
    Generates a list of tuples containing train and test indexes for cross-validation in panel data.

    The cross-validation strategy involves training the model with a specified number of months for all individuals
    and testing it on one month, with a gap between the training and test sets. The indexes are provided for each 
    split, allowing for time-series cross-validation.

    Args:
        data (pd.DataFrame): The input DataFrame containing the panel data.
        splits (int): The number of cross-validation splits.
        date_column (str): The name of the column containing date information.
        months_gap (int): The gap (in months) between the training and test periods.
        months_test (int): The number of months in the test set.
        max_train_size (int, optional): The maximum size of the training set. If None, no limit is applied.
        undersampling_rate (int, optional): The rate for undersampling the majority class in the training set. 
                                             If None, no undersampling is performed.
        target_column (str, optional): The name of the target column used for undersampling. Required if 
                                        undersampling_rate is provided.

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: A list of tuples, where each tuple contains two numpy arrays:
                                             the first array contains train indexes and the second array
                                             contains test indexes.
    """
    # First, the unique values of months are saved in months.
    months = data[date_column].sort_values().unique()

    # Second, a TimeSeriesSplit object is instantiated.
    tscv=TimeSeriesSplit(gap=months_gap, 
                         n_splits=splits, 
                         test_size=months_test,
                         max_train_size=max_train_size)
    


    # Third, the indexes for each split are generated and saved in the 
    # final_indexes list
    final_indexes=[]
    # The splits are first performed in the months Series. This way,
    # the months for train and test are obtained.
    try:
        _ = tscv.split(months)
    except:
        breakpoint()

    for (month_train_index, month_test_index) in tscv.split(months):
        # The tscv returns indexes, to obtain the actual months of train and test
        # it is necessary to use the months Series.
        training_months=months[month_train_index]
        test_months=months[month_test_index]

        # Then, the data is filtered to keep only observations either in the training
        # or test period. The training data is undersampled according to the 
        if undersampling_rate:
            if target_column==None: 
                raise RuntimeError("The target_column parameter should be provided\
                                   when an undersampling_rate is passed.")

            # If an undersampling rate was specified, the majority class is undersampled
            # only for the training set.
            rus = RandomUnderSampler(sampling_strategy=undersampling_rate, random_state=42)

            train_data=data[data[date_column].isin(training_months)].copy()
            train_indexes=rus.fit_resample(
                pd.DataFrame(train_data.index, columns=['idx']),
                train_data[target_column])[0]['idx'].values
        else:
            train_indexes=data[data[date_column].isin(training_months)].index

        test_indexes=data[data[date_column].isin(test_months)].index

        # Finally, the resulting indexes are included as a tuple in the
        # final_indexes list.

        msg = f"train shape: ({train_indexes.shape}) - test shape: ({test_indexes.shape})"
        logger.info(msg)

        final_indexes.append((train_indexes, test_indexes))

    return final_indexes

def select_features(
        data: pd.DataFrame,
        universe_of_features: List,
        target: str,
        cv: Union[List, int],
        n_features_to_select: int,
        direction: str='forward',
        scoring: str='ks'
    ):
    """
    Selects the best features from a Pandas DataFrame using sequential feature selection.

    This function performs feature selection by evaluating the importance of each feature
    based on a specified scoring metric and cross-validation strategy.

    Args:
        data (pd.DataFrame): The input DataFrame containing the features and target column.
        universe_of_features (List[str]): A list of feature names to consider for selection.
        target (str): The name of the target column.
        cv (Union[List, int]): The cross-validation strategy or the number of folds for cross-validation.
        n_features_to_select (int): The number of top features to select.
        direction (str, optional): The direction of feature selection, either 'forward' or 'backward'. Defaults to 'forward'.
        scoring (str, optional): The scoring metric to use for feature selection. Defaults to 'ks'. 
                                 Other options are supported if they are compatible with the `make_scorer` function.

    Returns:
        List[str]: A list of selected feature names.
    """

    all_selected_features = []


    msg = f"target variable {target}"
    logger.info(msg)

    msg = f"data shape {data.shape}"
    logger.info(msg)

    cols_not_in_universe = sorted(set(data.columns) - set(universe_of_features))

    features_not_in_data = sorted(set(universe_of_features) - set(data.columns))

    msg = "\n - ".join(cols_not_in_universe)
    logger.info("Columns not in universe:")
    logger.info(msg)

    msg = "\n - ".join(features_not_in_data)
    logger.info("Features not in data")
    logger.info(msg)

    logger.info("Start mRMR selection process")

    for i in range(10):
        data_sample = data.sample(frac=0.01, random_state=i)
        X = data_sample[universe_of_features]
        y = data_sample[target]

        selected_features, relevance, redundancy = mrmr_classif(X=X, y=y, K=20, relevance="ks", return_scores=True)

        score = relevance[selected_features] / redundancy.loc[selected_features, selected_features].apply(np.mean, axis=1)

        print(score)

        all_selected_features += selected_features

    all_selected_features = sorted(set(all_selected_features))

    msg = "\n - ".join(all_selected_features)

    logger.info("Selected Features - mRMR")
    logger.info(msg)

    # breakpoint()

    model_hyperparameters = {
        "learning_rate": 0.1, #0.05
        # "max_depth": 5, #5
        "n_estimators": 100, #1500
        "colsample_bytree": 0.7, # 0.9
        # "num_leaves": 31, # 15
        # "min_data_in_leaf": 10, #500
        "is_unbalance": True,
        # "force_row_wise": True,
        "force_col_wise": True,
        # "num_threads": 5,
        "verbose": 0,
        # "lambda_l1": 1,
        # "lambda_l2": 1,
    }

    estimator=lgb.LGBMClassifier(**model_hyperparameters)

    if scoring=='ks':
        # As the ks metric is not natively included in scikitlearn, it should
        # be created separately.
        def ks_stat(y, yhat):
            return ks_2samp(yhat[y==1], yhat[y==0]).statistic
        ks = make_scorer(ks_stat, needs_proba=True)
        selector=SequentialFeatureSelector(
            estimator, 
            n_features_to_select=n_features_to_select,
            tol=0.05,
            direction=direction,
            scoring=ks,
            cv=5, #cv,
            # n_jobs=-1
        )
    else:
        selector=SequentialFeatureSelector(
            estimator, 
            n_features_to_select=n_features_to_select,
            direction=direction,
            scoring=scoring,
            cv=5, #cv,
            n_jobs=-1
        )

    # selector.fit(data[universe_of_features],
    #              data[target])

    data_sampled = data.sample(frac=0.01, random_state=42).copy()

    # X = data[universe_of_features]
    # y = data[target]

    logger.info("Starting feature selection")

    # selected_features, relevance, redundancy = mrmr_classif(X=X, y=y, K=60, relevance="ks", return_scores=True)
    # # selected_features, relevance, redundancy = mrmr_sp.mrmr_classif(X=X, y=y, K=60, relevance="ks", return_scores=True)

    # score = relevance[selected_features] / redundancy.loc[selected_features, selected_features].apply(np.mean, axis=1)

    # print(score)

    msg = f"Initial number of features: {len(all_selected_features)}"
    logger.info(msg)

    msg = f"Desired number of features: {n_features_to_select}"
    logger.info(msg)
 
    selector.fit(data_sampled[all_selected_features].values, data_sampled[target])

    support = selector.get_support()

    breakpoint()

    final_features = np.array(all_selected_features)[support].tolist()

    msg = "\n - ".join(final_features)

    logger.info("Selected Features - Sequential Selector")
    logger.info(msg)

    # print(selector.get_feature_names_out())
    # print(final_features)

    return final_features


def tune_hyperparameters(
        data: pd.DataFrame,
        parameters_distribution: Dict,
        cv: Union[List, int],
        features: List,
        target: str,
        iterations: int,
        scoring: List,
        metric_to_maximize: str,
):
    """
    Tunes hyperparameters for a ML classifier using randomized search and cross-validation.

    This function performs hyperparameter optimization by running a randomized search 
    over specified parameter distributions. It uses cross-validation to evaluate the 
    performance of different hyperparameter combinations and selects the best one based 
    on the specified metric.

    Args:
        data (pd.DataFrame): The input DataFrame containing the features and target column.
        parameters_distribution (Dict[str, Any]): A dictionary specifying the parameter grid 
                                                   to sample from during hyperparameter tuning.
        cv (Union[List, int]): The cross-validation strategy or the number of folds for cross-validation.
        features (List[str]): A list of feature names to use in the model.
        target (str): The name of the target column.
        iterations (int): The number of iterations to perform in the randomized search.
        scoring (List[str]): A list of scoring metrics to evaluate during cross-validation. 
                             The 'ks' metric can be included and will be computed separately.
        metric_to_maximize (str): The metric to maximize for selecting the best model.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'tuning_results': A DataFrame with the results of the hyperparameter tuning process.
            - 'tuning_parameters': The best hyperparameter combination found.
            - 'tuning_estimator': The best estimator found with the hyperparameters.
    """
    estimator=lgb.LGBMClassifier()

    if 'ks' in scoring:

        # As the ks metric is not natively included in scikitlearn, it should
        # be created separately.
        def ks_stat(y, yhat):
            return ks_2samp(yhat[y==1], yhat[y==0]).statistic
        ks = make_scorer(ks_stat, needs_proba=True)

        # When a callable is one of the scorings, it should be included in a
        # dictionary.
        scoring_with_ks={}
        for i in scoring:
            if i!='ks':
                scoring_with_ks[i]=i
            else:
                scoring_with_ks[i]=ks
        if metric_to_maximize=='ks':
            clf=RandomizedSearchCV(estimator, 
                               param_distributions=parameters_distribution, 
                               n_iter=iterations, 
                               cv=cv, 
                               scoring=scoring_with_ks, 
                               refit='ks',
                               verbose=2,
                               n_jobs=-1)
        else:
            clf=RandomizedSearchCV(estimator, 
                    param_distributions=parameters_distribution, 
                    n_iter=iterations, 
                    cv=cv, 
                    scoring=scoring_with_ks, 
                    refit=metric_to_maximize,
                    verbose=2,
                    n_jobs=-1)
    else:
        clf=RandomizedSearchCV(estimator, 
                           param_distributions=parameters_distribution, 
                           n_iter=iterations, 
                           cv=cv, 
                           scoring=scoring, 
                           refit=metric_to_maximize,
                           verbose=2,
                           n_jobs=-1)
    clf.fit(data[features], data[target])
    print("----------------------")
    print('Metric to maximize: ', metric_to_maximize)
    mean_metric_col='mean_test_'+metric_to_maximize
    print('Value of max metric: ', pd.DataFrame(clf.cv_results_)[mean_metric_col].max())
    print("----------------------")

    results_df = pd.DataFrame(clf.cv_results_)
    results_columns = [col for col in results_df.columns if ("param_" in col) or ("mean_test_" in col)]
    print(results_df[results_columns].sort_values("mean_test_recall", ascending=False).head().to_markdown(index=False))
    print(clf.best_params_)


    return {'tuning_results':pd.DataFrame(clf.cv_results_),
            'tuning_parameters': clf.best_params_,
            'tuning_estimator': clf.best_estimator_}

def cv_function(data: pd.DataFrame,
                cv_splits: np.ndarray,
                X_columns: List, 
                threshold: float, 
                target_column: str, 
                hyperparameters: Dict):
    """
    Performs cross-validation on the given data using ML classifier, evaluating performance metrics.

    Args:
        data (pd.DataFrame): The dataset containing features and target variable.
        cv_splits (np.ndarray): Array of train-test index splits for cross-validation.
        X_columns (List[str]): List of column names to be used as features.
        threshold (float): Decision threshold for classification.
        target_column (str): Name of the column in `data` that contains the target variable.
        hyperparameters (Dict[str, any]): Dictionary of hyperparameters for ML classifier.

    Returns:
        tuple:
            - resultados (dict): Dictionary with mean values of performance metrics across cross-validation folds.
            - results_tscv (pd.DataFrame): DataFrame containing performance metrics
    """
    kss=[]
    ksp=[]
    precisions=[]
    recalls=[]
    roc=[]
    pr=[]
    tps=[]
    tns=[]
    fps=[]
    fns=[]
    threshold=threshold
    period_train=[]
    period_test=[]

    # cv_splits.reversed()

    # for (train_index, test_index) in tqdm.tqdm(cv_splits[::-1]):
    #     training_months=pd.Series(list(set(data.iloc[train_index]._observ_end_dt)))
    #     training_period=f"{training_months.min()}/{training_months.max()}"

    #     test_months=pd.Series(list(set(data.iloc[test_index]._observ_end_dt)))
    #     testing_period=f"{test_months.min()}/{test_months.max()}"

    #     print(training_period, testing_period)

    # breakpoint()
 
    for (train_index, test_index) in tqdm.tqdm(cv_splits[::-1]):

        training_months=pd.Series(list(set(data.iloc[train_index]._observ_end_dt)))
        training_period=f"{training_months.min()}/{training_months.max()}"
        period_train.append(training_period)

        test_months=pd.Series(list(set(data.iloc[test_index]._observ_end_dt)))
        testing_period=f"{test_months.min()}/{test_months.max()}"

        if len(set(training_months).intersection(set(test_months)))>0:
            print('error')

        period_test.append(testing_period)

        X_train_cv=data.iloc[train_index][X_columns].copy()
        y_train_cv=data.iloc[train_index][target_column].copy()
        X_test_cv=data.iloc[test_index][X_columns].copy()
        y_test_cv=data.iloc[test_index][target_column].copy()

        lgb_classifier = lgb.LGBMClassifier(**hyperparameters)
        lgb_classifier.fit(X_train_cv, y_train_cv)
        y_proba = lgb_classifier.predict_proba(X_test_cv)[:,-1]
        y_pred = np.where(y_proba > threshold, 1 ,0)

        #Precision and recall
        precisions.append(precision_score(y_test_cv, y_pred, zero_division=1))
        recalls.append(recall_score(y_test_cv, y_pred))

        #PR AUC
        precision, recall, _ = precision_recall_curve(y_test_cv, y_proba)
        aux=pd.DataFrame()
        aux['precision']=precision
        aux['recall']=recall
        aux.sort_values('recall', inplace=True)
        pr.append(auc(aux['recall'], aux['precision']))

        #ROC AUC
        try:
            auc_score = roc_auc_score(y_test_cv, y_proba)
        except:
           auc_score = 0.0
        roc.append(auc_score)

        # KS
        aux_ks=pd.DataFrame()
        aux_ks['real']=y_test_cv
        aux_ks['pred']=y_proba
        kss.append(ks_2samp(aux_ks[aux_ks.real==0].pred, aux_ks[aux_ks.real==1].pred).statistic)
        ksp.append(ks_2samp(aux_ks[aux_ks.real==0].pred, aux_ks[aux_ks.real==1].pred).pvalue)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test_cv, y_pred).ravel()
        tps.append(tp)
        tns.append(tn)
        fps.append(fp)
        fns.append(fn)

    results_tscv=pd.DataFrame()
    results_tscv['iteration']=range(1, len(period_train)+1)
    results_tscv['train_period']=period_train
    results_tscv['test_period']=period_test
    results_tscv['precision']=precisions
    results_tscv['recall']=recalls
    results_tscv['roc_auc']=roc
    results_tscv['pr_auc']=pr
    results_tscv['ks']=kss
    results_tscv['ks-pvalue']=ksp
    results_tscv['tp']=tps
    results_tscv['tn']=tns
    results_tscv['fp']=fps
    results_tscv['fn']=fns
    resultados=dict(results_tscv[['precision', 'recall', 'roc_auc', 'pr_auc', 'ks', 'tp', 'tn', 'fp', 'fn']].mean())

    return resultados, results_tscv


def get_explainer(model_clf, X, sample=0.02):

    explainer = shap.TreeExplainer(model_clf, data=X, model_output="probability")
    return explainer


def train_model(
        data: pd.DataFrame,
        columns: List,
        model_hyperparameters: Dict,
        threshold: float,
        target_column: str,
        cv_indexes: np.ndarray,
):
    """
    Trains a ML model and evaluates its performance using cross-validation.

    This function uses the provided `cv_function` to perform cross-validation and obtain metrics. It then trains
    a ML model on the entire dataset using the specified hyperparameters and returns the evaluation metrics,
    the metrics DataFrame, and the trained model.

    Args:
        data (pd.DataFrame): The dataset containing features and target variable.
        columns (List[str]): List of column names to be used as features.
        model_hyperparameters (Dict[str, any]): Dictionary of hyperparameters for ML classifier.
        threshold (float): Decision threshold for classification.
        target_column (str): Name of the column in `data` that contains the target variable.
        cv_indexes (np.ndarray): Array of train-test index splits for cross-validation.

    Returns:
        tuple:
            - metrics (dict): Dictionary with mean values of performance metrics from cross-validation.
            - metrics_df (pd.DataFrame): DataFrame containing performance metrics for each fold.
            - classifier : Trained classifier.
    """

    model_hyperparameters = {
        "learning_rate": 0.1, #0.05
        # "max_depth": 5, #5
        "n_estimators": 100, #1500
        "colsample_bytree": 0.7, # 0.9
        # "num_leaves": 31, # 15
        # "min_data_in_leaf": 1000, #500
        # "is_unbalance": True,
        "force_row_wise": True,
        # "force_col_wise": True,
        # "num_threads": 5,
        "verbose": 1,
        # "lambda_l1": 1,
        # "lambda_l2": 1,
    }

    # breakpoint()

    # data["cluster_id"] = data["cluster_id"].str.split("_id_", expand=True)[1].astype(int)

    logger.info("Starting cross validation training")

    msg = f"data shape: {data.shape}"
    logger.info(msg)

    metrics, metrics_df = cv_function(
        data=data,
        cv_splits=cv_indexes,
        X_columns=columns,
        threshold=threshold,
        target_column=target_column,
        hyperparameters=model_hyperparameters
    )

    print(metrics)

    print(metrics_df.to_markdown())

    X_train=data[columns].copy()
    y_train=data[target_column].copy()

    logger.info("Starting final training")

    lgb_classifier = lgb.LGBMClassifier(**model_hyperparameters)
    lgb_classifier.fit(X_train, y_train)

    explainer = get_explainer(lgb_classifier, X_train, sample=0.02)

    return metrics, metrics_df, lgb_classifier, explainer


def validate_model(
        data: pd.DataFrame,
        X_columns: List[str],
        model: Any,
        target_column: str,
        threshold: float = 0.5,
):
    """
    Performs cross-validation on the given data using ML classifier, evaluating performance metrics.

    Args:
        data (pd.DataFrame): The dataset containing features and target variable.
        cv_splits (np.ndarray): Array of train-test index splits for cross-validation.
        X_columns (List[str]): List of column names to be used as features.
        threshold (float): Decision threshold for classification.
        target_column (str): Name of the column in `data` that contains the target variable.
        hyperparameters (Dict[str, any]): Dictionary of hyperparameters for ML classifier.

    Returns:
        tuple:
            - resultados (dict): Dictionary with mean values of performance metrics across cross-validation folds.
            - results_tscv (pd.DataFrame): DataFrame containing performance metrics
    """
    kss=[]
    ksp=[]
    precisions=[]
    recalls=[]
    roc=[]
    pr=[]
    tps=[]
    tns=[]
    fps=[]
    fns=[]
    threshold=threshold
    period_train=[]
    period_test=[]

    # cv_splits.reversed()

    # for (train_index, test_index) in tqdm.tqdm(cv_splits[::-1]):
    #     training_months=pd.Series(list(set(data.iloc[train_index]._observ_end_dt)))
    #     training_period=f"{training_months.min()}/{training_months.max()}"

    #     test_months=pd.Series(list(set(data.iloc[test_index]._observ_end_dt)))
    #     testing_period=f"{test_months.min()}/{test_months.max()}"

    #     print(training_period, testing_period)

    # breakpoint()

    unique_periods = data["_observ_end_dt"].unique()
 
    for time_period in tqdm.tqdm(unique_periods):

        test_index = data[data["_observ_end_dt"] == time_period].index

        period_test.append(time_period)

        X_test_cv = data.iloc[test_index][X_columns].copy()
        y_test_cv = data.iloc[test_index][target_column].copy()
 
        y_proba = model.predict_proba(X_test_cv)[:,-1]
        y_pred = np.where(y_proba > threshold, 1, 0)
 
        #Precision and recall
        precisions.append(precision_score(y_test_cv, y_pred, zero_division=1))
        recalls.append(recall_score(y_test_cv, y_pred))
 
        #PR AUC
        precision, recall, _ = precision_recall_curve(y_test_cv, y_proba)
        aux=pd.DataFrame()
        aux['precision'] = precision
        aux['recall'] = recall
        aux.sort_values('recall', inplace=True)
        pr.append(auc(aux['recall'], aux['precision']))
 
        #ROC AUC
        try:
            auc_score = roc_auc_score(y_test_cv, y_proba)
        except:
           auc_score = 0.0
        roc.append(auc_score)
 
        # KS
        aux_ks=pd.DataFrame()
        aux_ks['real']=y_test_cv
        aux_ks['pred']=y_proba
        kss.append(ks_2samp(aux_ks[aux_ks.real==0].pred, aux_ks[aux_ks.real==1].pred).statistic)
        ksp.append(ks_2samp(aux_ks[aux_ks.real==0].pred, aux_ks[aux_ks.real==1].pred).pvalue)
 
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test_cv, y_pred).ravel()
        tps.append(tp)
        tns.append(tn)
        fps.append(fp)
        fns.append(fn)
 
    results_tscv=pd.DataFrame()
    results_tscv['iteration']=range(1, len(period_test)+1)
    results_tscv['test_period']=period_test
    results_tscv['precision']=precisions
    results_tscv['recall']=recalls
    results_tscv['roc_auc']=roc
    results_tscv['pr_auc']=pr
    results_tscv['ks']=kss
    results_tscv['ks-pvalue']=ksp
    results_tscv['tp']=tps
    results_tscv['tn']=tns
    results_tscv['fp']=fps
    results_tscv['fn']=fns
    results_tscv = results_tscv.sort_values("test_period")

    resultados=dict(results_tscv[['precision', 'recall', 'roc_auc', 'pr_auc', 'ks', 'tp', 'tn', 'fp', 'fn']].mean())

    return resultados, results_tscv


def report_features(
        data: pd.DataFrame,
        columns: List,
        explainer: Any,
):

    X = data.sample(frac=0.01)
    shap_values = explainer(X[columns])

    beeswarm_plot = plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.close()

    average_importance_plot = plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.close()

    shap_df = pd.DataFrame(shap_values.values, columns=columns)

    positive_feature_importance = (
        shap_df[shap_df > 0]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={'index':'feature',0:'weight'})
        .head(10)
    )

    negative_feature_importance = (
        shap_df[shap_df < 0]
        .mean()
        .sort_values(ascending=True)
        .reset_index()
        .rename(columns={'index':'feature',0:'weight'})
        .head(10)
    )

    feature_importance = (
        shap_df
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={'index':'feature',0:'weight'})
    )

    print("\n")
    logger.info("General Feature Importance")
    msg = feature_importance.head(10).to_markdown()
    logger.info(msg)

    print("\n")
    logger.info("Positive Feature Importance")
    msg = positive_feature_importance.to_markdown()
    logger.info(msg)

    print("\n")
    logger.info("Negative Feature Importance")
    msg = negative_feature_importance.to_markdown()
    logger.info(msg)

    return beeswarm_plot, average_importance_plot, feature_importance


def validation_node(
        data: pd.DataFrame,
        columns: List,
        model: Any,
        explainer: Any,
        target_column: str,
        threshold: float
):

    msg = f"dates: ({data["_observ_end_dt"].min()}) - ({data["_observ_end_dt"].max()})"
    logger.info(msg)

    metrics, metrics_df = validate_model(data, columns, model, target_column)

    print(metrics)
    print(metrics_df.to_markdown())

    # for expr in ["is_new_joiner == 1", "is_uncommited == 1", "is_potential_loyal == 1", "is_loyal == 1", "is_lost == 1", "is_gone == 1"]:
    #     logger.info(expr)
    #     metrics, metrics_df = validate_model(data, columns, model, target_column, expr)

    #     print(metrics)
    #     print(metrics_df.to_markdown())


    beeswarm_plot, average_importance_plot, feature_importance = report_features(data, columns, explainer)

    return metrics, metrics_df, beeswarm_plot, average_importance_plot, feature_importance
