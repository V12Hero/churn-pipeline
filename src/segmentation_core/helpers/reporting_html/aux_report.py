# (c) McKinsey & Company 2016 – Present
# All rights reserved
#
#
# This material is intended solely for your internal use and may not be reproduced,
# disclosed or distributed without McKinsey & Company's express prior written consent.
# Except as otherwise stated, the Deliverables are provided ‘as is’, without any express
# or implied warranty, and McKinsey shall not be obligated to maintain, support, host,
# update, or correct the Deliverables. Client guarantees that McKinsey’s use of
# information provided by Client as authorised herein will not violate any law
# or contractual right of a third party. Client is responsible for the operation
# and security of its operating environment. Client is responsible for performing final
# testing (including security testing and assessment) of the code, model validation,
# and final implementation of any model in a production environment. McKinsey is not
# liable for modifications made to Deliverables by anyone other than McKinsey
# personnel, (ii) for use of any Deliverables in a live production environment or
# (iii) for use of the Deliverables by third parties; or
# (iv) the use of the Deliverables for a purpose other than the intended use
# case covered by the agreement with the Client.
# Client warrants that it will not use the Deliverables in a "closed-loop" system,
# including where no Client employee or agent is materially involved in implementing
# the Deliverables and/or insights derived from the Deliverables.

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from IPython.display import display
from plotly import graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score


def plotly_hist(
    series: Dict,
    width: int = None,
    height: int = None,
    title: str = None,
    histnorm: str = None,
):
    """Creates a Plotly Histogram.

    Creates a Plotly Histogram from a dict of pandas series

    args:
        series: dict of pandas series {'name': pd.Series}
        width: (optional) width of the figure
        height: (optional) height of the figure
        Title: (optional) Title of the figure

    returns:
        fig: plotly figure
    """
    if not width:
        width = 800
    if not height:
        height = 400
    if not title:
        title = "Histogram"

    fig = go.Figure()

    # Iterating over dictionary items:
    for name, serie in series.items():
        fig.add_trace(
            go.Histogram(
                x=serie,
                name=f"{serie.name}-{name}",
                # marker_color="#388297",
                opacity=0.6,
                histnorm=histnorm,
            )
        )
    fig.update_layout(
        width=width, height=height, title_text=f"<b>{title}</b>", barmode="overlay"
    )

    return fig


def plotly_pred_act(
    act: pd.Series,
    pred: pd.Series,
    title: str,
    mode: str = None,
    cuts: list = None,
    gaps: bool = None,
    height: int = None,
    width: int = None,
):
    """
    Creates a plotly figure comparing between predicted and actual
    args:
        act: Pandas Series of actual
        pred: Pandas Series of predicted
        title: (optional) Title of the figure
        mode: (optional) mode of the trace possible values ['lines', 'markers', 'lines+markers']
        cuts: (optional) list of ints to make traces
        gaps: (optional) boolean: True if render with gaps
        height: (optional) height of the figure
        width: (optional) width of the figure
    returns:
        fig: plotly figure
    """
    if not mode:
        mode = "lines"
    if not gaps:
        gaps = True
    if not width:
        width = 800
    if not height:
        height = 400

    if gaps:
        gaps_df = create_gaps(act, pred)
        act = gaps_df["act"]
        pred = gaps_df["pred"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=act.index,
            y=act,
            name="act",
            mode=mode,
            marker=dict(color="#1f77b4", size=4),
            line=dict(width=1.5),
            opacity=0.9,
            connectgaps=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=pred.index,
            y=pred,
            name="pred",
            mode=mode,
            marker=dict(color="#ff7f0e", size=4),
            line=dict(width=1.3),
            opacity=0.9,
            connectgaps=False,
        )
    )

    if cuts:
        for cut in cuts:
            fig.add_shape(
                go.layout.Shape(
                    type="line",
                    x0=min(act.index),
                    y0=cut,
                    x1=max(act.index),
                    y1=cut,
                    line=dict(color="#333333", width=1, dash="dash"),
                )
            )

    fig.update_layout(
        width=width,
        height=height,
        template="plotly_white",
        title_text=f"<b>{title}</b>",
    )
    fig.show()


def create_gaps(act: pd.Series, pred: pd.Series):
    """
    Creates a dataframe with nans between gaps
    args:
        act: Pandas Series of actual
        pred: Pandas Series of predicted
    returns:
        df2: plotly figure
    """
    df = pd.DataFrame(act)
    df.columns = ["act"]
    df["pred"] = pred

    dates = pd.date_range(min(df.index), max(df.index), freq="4h")

    df2 = pd.DataFrame(dates)
    df2.columns = ["timestamp"]

    df2 = pd.merge(df2, df.reset_index(), how="left").set_index("timestamp")

    return df2


def show_perf_metrics(y_true: pd.Series, y_pred: pd.Series, set: str):
    """
    creates performance metrics (used in baseline models)
    args:
        y_true: pd.Series of true values
        y_pred: pd.Series of predicted values
        set: str, name of the set
    returns:
        metrics: pd.DataFrame with the performance metrics
    """

    r2 = r2_score(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** (1 / 2)

    metrics = pd.DataFrame(
        zip([set], [r2], [mape_val], [mse], [rmse]),
        columns=["set", "r2", "mape", "mse", "rmse"],
    )

    return metrics


def mape(y_true: pd.Series, y_pred: pd.Series):
    """
    returns the mean average percentage error
    args:
        y_true: pd.Series of true values
        y_pred: pd.Series of predicted values
    returns:
        mape: mean average percentage error
    """
    out = 100 * (np.abs(y_true - y_pred) / y_true)
    mape = out.mean()
    return mape


def create_mean_model(y_tr: pd.Series, y_ts: pd.Series):
    """
    creates a mean baseline model
    args:
        y_tr: pd.Series of train set response variable
        y_ts: pd.Series of test set response variable
    returns:
        metrics: pd.DataFrame with the performance metrics of the mean baseline model
    """
    y_train_mean = y_tr.mean()
    predict = np.ones(len(y_ts)) * y_train_mean
    df_baseline = pd.DataFrame({"y_test": y_ts, "mean_pred": predict})

    plotly_pred_act(
        df_baseline["y_test"], df_baseline["mean_pred"], "Baseline mean model"
    )

    metrics = show_perf_metrics(
        df_baseline["y_test"], df_baseline["mean_pred"], "mean-test"
    )

    return metrics


def create_shift_model(y_ts: pd.Series):
    """
    creates a shift baseline model
    args:
        y_ts: pd.Series of test set response variable
    returns:
        metrics: pd.DataFrame with the performance metrics of the shift baseline model
    """

    predict = y_ts.shift(1)
    df_baseline = pd.DataFrame({"y_test": y_ts, "prev_shift": predict}).iloc[1:, :]

    plotly_pred_act(
        df_baseline["y_test"], df_baseline["prev_shift"], "Baseline shift model"
    )

    metrics = show_perf_metrics(
        df_baseline["y_test"], df_baseline["prev_shift"], "shift-test"
    )
    return metrics


def display_baseline_models(y_tr: pd.Series, y_ts: pd.Series):
    """
    display the baseline models performance ()
    args:
        y_tr: pd.Series of train set response variable
        y_ts: pd.Series of test set response variable
    """
    display(create_mean_model(y_tr, y_ts))
    display(create_shift_model(y_ts))
    y = y_tr.append(y_ts)
    var_coef = np.std(y) / np.mean(y)
    print("-" * 40)
    print(f"coef_var-test = {var_coef}")
    print("-" * 40)


def shap_importance(trained_model, data: pd.DataFrame, feat_cols: list, td, disp: int):
    """
    display the shap importance plot and return the shap values
    args:
        trained_model: sklearn Pipeline with a regressor step
        data: pd.DataFrame of data to calculate shap values
        feat_cols: list of columns
        td: TagDict
        displ: int how many variables display
    """
    trained_model = trained_model["regressor"]

    data = data[feat_cols]

    explainer = shap.TreeExplainer(trained_model)
    shap_values = explainer.shap_values(data)
    shap.summary_plot(
        shap_values,
        data,
        feature_names=[td.description(feat) for feat in feat_cols],
        max_display=disp,
    )

    df_fi = pd.DataFrame(
        index=feat_cols, data=np.mean(np.abs(shap_values), axis=0), columns=["value"]
    )
    df_fi["description"] = [td.description(feat) for feat in df_fi.index]

    df_sorted = df_fi.sort_values(by="value", ascending=False)
    df_sorted["order"] = df_sorted.reset_index().index
    df_fi = df_fi.merge(df_sorted.order, how="left", left_index=True, right_index=True)

    return df_fi


def plot_heatmap(data, mask_par: bool = None):
    """plot the correlation map.

    args:
        data: pd.Dataframe of pairwise correlations
        mask_par: (optional) mask the superior triangle, default: True
    """
    if not mask_par:
        mask_par = True

    mask = np.zeros_like(data, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Want diagonal elements as well
    mask[np.diag_indices_from(mask)] = False

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(30, 30))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(10, 220, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio

    if mask_par:
        fig = sns.heatmap(
            data,
            mask=mask,
            cmap=cmap,
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
            annot=True,
            fmt=".2f",
        )
    else:
        fig = sns.heatmap(
            data,
            cmap=cmap,
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
            annot=True,
            fmt=".2f",
        )
    # fig.get_figure().savefig("heatmap.png")
    fig


def plotly_scatter(
    df,
    vars,
    rng=None,
    width=900,
    height=300,
    title="Scatterplot",
    mode="markers",
    xmin="2020-01",
):
    df = df[df.index > xmin]

    fig = go.Figure()

    for var in vars:
        fig.add_trace(
            go.Scatter(
                mode=mode,
                x=df.index,
                y=df[var],
                name=var,
                # marker_color="#388297",
                opacity=0.6,
                connectgaps=False,
            )
        )
    if rng:
        fig.update_layout(yaxis=dict(range=rng))
    fig.update_layout(width=width, height=height, title_text=f"<b>{title}</b>")

    return fig
