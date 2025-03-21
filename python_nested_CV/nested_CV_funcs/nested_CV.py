import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import root_mean_squared_error, mean_squared_error, median_absolute_error
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from tqdm.auto import tqdm

DAYS_PER_MONTH = 30.4
DAYS_PER_YEAR = 365.25

def generate_performance_plots(
    y_pred: pd.Series, y_actual: pd.Series, performance_dict: dict, method_name: str
):
    """
    Creates plots for model testing script, showing:
    - scatterplot of predicted vs actual age
    - summary statistics
    - scatterplot of residuals vs actual age
    - histogram of residuals
    - histogram of the absolute value of residuals
    - CDF plot of the absolute value of residuals
    Parameters
    ----------
    y_pred: pd.Series
        Model-predicted ages in days
    y_actual: pd.Series
        Actual ages in days
    performance_dict: dict
        Dictionary of performance metrics
    method_name: str
        Type of model built

    Returns
    -------
    plots
    """
    _y_pred = y_pred
    _y_actual = y_actual
    residual = _y_pred - _y_actual

    fig, (ax1, ax2) = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(8)

    sns.scatterplot(
        x=_y_actual / DAYS_PER_YEAR, y=_y_pred / DAYS_PER_YEAR, ax=ax1[0], s=10
    )
    ax1[0].axline((0, 0), slope=1, c="k", ls="--")
    ax1[0].set_xlabel("Actual age (years)")
    ax1[0].set_ylabel("Predicted age (years)")

    ax1[1].set_axis_off()
    rmse_str = f"{performance_dict['rmse'] / DAYS_PER_MONTH:.2f} months"
    # param_str = str(
    #     {key: f"{value:.4f}"
    #      for key, value in performance_dict["best_params"].items()}
    # )
    ax1[1].table(
        cellText=[[rmse_str]],
        rowLabels=["RMSE"],
        loc="center",
        cellLoc="left",
        edges="open",
        colWidths=[1],
        rowLoc="right",
    )

    sns.scatterplot(
        x=_y_actual / DAYS_PER_YEAR, y=residual / DAYS_PER_MONTH, ax=ax1[2], s=10
    )
    ax1[2].axline((0, 0), slope=0, c="k", ls="--")
    ax1[2].set_xlabel("Actual age (years)")
    ax1[2].set_ylabel("Predicted-actual age (months)")

    sns.histplot(x=residual / DAYS_PER_MONTH, ax=ax2[0], binwidth=3, stat="proportion")
    ax2[0].set_xlabel("Predicted-actual age (months)")

    sns.histplot(
        x=abs(residual / DAYS_PER_MONTH),
        ax=ax2[1],
        binrange=(0, 50),
        binwidth=1,
        stat="proportion",
    )
    ax2[1].set_xlim(-1, 50)
    ax2[1].set_xticks(range(0, 51, 6))
    ax2[1].set_xlabel("Absolute predicted-actual age (months)")

    sns.ecdfplot(
        x=abs(residual / DAYS_PER_MONTH),
        ax=ax2[2],
    )
    ax2[2].set_xlim(0, 50)
    ax2[2].set_xticks(range(0, 51, 6))
    ax2[2].set_xlabel("Absolute predicted-actual age (months)")

    fig.suptitle(f"{method_name} on {len(y_pred)} holdout samples", fontsize="xx-large")
    plt.tight_layout()

def hyperparameter_tuning(
    method_dict: dict,
    _X: pd.DataFrame,
    _y: pd.Series,
    outer_fold_groups: pd.Series,
    _n_inner_folds: int = 2,
    _scoring: tuple = ("neg_root_mean_squared_error",),
    n_jobs: int = -1,
) -> object:
    """

     Parameters
     ----------
     method_name: str
         Type of model built
     method_dict: dict
         Hyperparameters to be explored
     _scoring: tuple
         Scoring metrics used during model training
         (Default value = ("neg_root_mean_squared_error", )
     _X: pd.DataFrame
         Feature matrix for training dataset
     _y: pd.Series
         Actual ages of samples in days for training dataset
     outer_fold_groups: pd.Series
         Group ID for each sample in training dataset
         [specifies fold for outer-fold cross-validation]
    _n_inner_folds: int
         Specifies number of inner folds for inner fold cross-validation
     n_jobs: int
         number processors to use (Default value = -1)

    Returns
     -------
         GridSearchCV.best_estimator_
    """
    ## Configure an iterator to loop over outer folds
    logo = LeaveOneGroupOut()
    outer_fold_indices = logo.split(_X, _y, outer_fold_groups)
    outer_results = []
    inner_results = []

    for i, (train_indices, test_indices) in tqdm(
            enumerate(outer_fold_indices),
            total=logo.get_n_splits(groups=outer_fold_groups)
    ):
        ## Extract the train and test data for this outer fold
        X_train, X_test, y_train, y_test = (
            _X.iloc[train_indices],
            _X.iloc[test_indices],
            _y.iloc[train_indices],
            _y.iloc[test_indices],
        )
        ## Define GridSearchCV object
        gridsearch = GridSearchCV(
            estimator=method_dict["estimator"],
            param_grid=method_dict["param_grid"],
            scoring=_scoring,
            refit=_scoring[0],
            cv=_n_inner_folds,
            return_train_score=True,
            n_jobs=n_jobs,
        )

        ## Perform grid search to get the hyperparamer set
        ## that produced the lowest mean RMSE across the inner folds,
        ## and fit a model on this outer fold's entire training
        ## subset using that hyperparameter set
        gridsearch.fit(X_train, y_train)
        ## Use this model to get predicted ages for this outer fold's test subset
        ## and calculate its inner-fold cross-validated RMSE
        y_pred = gridsearch.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        inner_fold_df = pd.DataFrame(gridsearch.cv_results_)
        inner_fold_df["outer_fold"] = i + 1
        inner_results.append(inner_fold_df)
        outer_results.append(
            {
                "outer_fold": i + 1,
                "best_params": gridsearch.best_params_,
                "rmse": rmse / DAYS_PER_MONTH,
            }
        )
    return pd.DataFrame(outer_results), pd.concat(inner_results)


def train_model(
    method_name: str,
    method_dict: dict,
    _X: pd.DataFrame,
    _y: pd.Series,
    _groups: pd.Series,
    _scoring: tuple = ("neg_root_mean_squared_error",),
    n_jobs: int = -1,
    _save_output: bool = False,
    prefix: str = False,
) -> object:
    """

     Parameters
     ----------
     method_name: str
         Type of model built
     method_dict: dict
         Hyperparameters to be explored
     _scoring: tuple
         Scoring metrics used during model training
         (Default value = ("neg_root_mean_squared_error", )
     _X: pd.DataFrame
         Feature matrix for training dataset
     _y: pd.Series
         Actual ages of samples in days for training dataset
     _groups: pd.Series
         Group ID for each sample in training dataset
         [specifies fold for cross-validation]
     n_jobs: int
         number processors to use (Default value = -1)
      _save_output: bool
         Whether to save outputs (Default value = False)
     prefix: str :
         If saving outputs, location to use (Default value = False)

    Returns
     -------
         GridSearchCV.best_estimator_
    """
    print(f"Selecting hyperparameters for {method_name} on training data")
    logo = LeaveOneGroupOut()
    gridsearch = GridSearchCV(
        estimator=method_dict["estimator"],
        param_grid=method_dict["param_grid"],
        scoring=_scoring,
        refit=_scoring[0],
        cv=logo.split(_X, _y, _groups),
        return_train_score=True,
        n_jobs=n_jobs,
    )
    gridsearch.fit(_X, _y)
    if _save_output:
        pd.DataFrame(gridsearch.cv_results_).to_csv(
            f"{prefix}/{method_name}.cv_results.csv"
        )
        pd.to_pickle(gridsearch.best_estimator_, 
            f"{prefix}/{method_name}.estimator.pkl"
        )

    return gridsearch.best_estimator_


def test_model(
    best_estimator: object,
    method_name: str,
    _X_holdout: pd.DataFrame,
    _y_holdout: pd.Series,
    _quantiles: np.ndarray = np.arange(0, 1.01, 0.05),
    _thresholds: np.ndarray = np.arange(0, 49, 3),
    _generate_plots: bool = False,
):
    """
    Predicts age of hold-out data an calculates several summary statistics
    on the residual error.

    Parameters
    ----------
    best_estimator: GridSearchCV.best_estimator_
        Best model generated during training
    method_name: str
        Name of the model being tested
    _X_holdout: pd.DataFrame
        Feature matrix of holdout dataset
    _y_holdout: pd.Series
        Actual age of holdout dataset (days)
    _quantiles: np.ndarray
        Percentiles at which to report residual error
        (Default value = np.arange(0, 1.01, 0.05)
    _thresholds: np.ndarray
        Range months to use when calcuating residual error
        (Default value = np.arange(0, 49, 3)
    _generate_plots: bool
        If summary plots should be created (Default value = False)

    Returns
    -------
    performance_df: pd.DataFrame
        Summary of overall model performance
    re_by_quant: pd.DataFrame
        Summary of residual error by specified percentiles
    quantile_by_re: pd.DataFrame
        Summary of residual error by specified month thresholds
    predictions_df: pd.DataFrame
        Age predictions in days for holdout set
    """

    print(f"Evaluating {method_name} with chosen hyperparameters on test data")
    _y_pred = best_estimator.predict(_X_holdout)
    _rmse = root_mean_squared_error(_y_holdout, _y_pred)
    _mae = median_absolute_error(_y_holdout, _y_pred)
    _resids = abs(_y_pred - _y_holdout) / DAYS_PER_MONTH

    performance_df = {
        "rmse": _rmse,
        "median_abs_error": _mae,
        "best_params": [best_estimator.get_params()],
    }

    re_by_quant = pd.DataFrame(
        {
            "method_name": method_name,
            "residual_error_months": _resids.quantile(_quantiles).reset_index(
                drop=True
            ),
            "quantile": _quantiles,
        }
    )

    quantile_by_re = pd.DataFrame(
        {
            "method_name": method_name,
            "residual_error_months": _thresholds,
            "quantile": [(_resids < x).sum() / len(_resids) for x in _thresholds],
        }
    )

    predictions_df = pd.DataFrame(
        {
            "method_name": method_name,
            "y_pred": _y_pred,
            "y_actual": _y_holdout,
        }
    )

    if _generate_plots:
        generate_performance_plots(_y_pred, _y_holdout, performance_df, method_name)

    return performance_df, re_by_quant, quantile_by_re, predictions_df


def tune_test_model(
    _X: pd.DataFrame,
    _y: pd.Series,
    _groups: pd.Series,
    _X_holdout: pd.DataFrame,
    _y_holdout: pd.Series,
    method_dicts: dict,
    n_jobs: int = -1,
    _scoring: tuple = ("neg_root_mean_squared_error",),
    _quantiles: np.ndarray = np.arange(0, 1.01, 0.05),
    _thresholds: np.ndarray = np.arange(0, 49, 3),
    _save_output: bool = False,
    _generate_plots: bool = False,
    prefix: str = False,
) -> tuple:
    """
    Trains then evaluates model(s) specified in the (method_dicts). Function
    will loop through each method provided and using GridSearchCV() to
    explore each hyperparam, using X cross-fold validation. X in this case is
    the number of unique groups provided by _groups. For each method, model
    performance is evaluated using an independent hold out dataset (the
    _holdout arguements) and a set of standard performance metric tables are
    returned.

    Parameters
    ----------
    _X: pd.DataFrame
        Feature matrix for training dataset
    _y: pd.Series
        Actual ages of samples in days for training dataset
    _groups: pd.Series
        Group ID for each sample in training dataset
        [specifies fold for cross-validation]
    _X_holdout: pd.DataFrame
        Feature matrix for holdout dataset
    _y_holdout: pd.Series
        Actual ages of samples in days for holdout dataset
    method_dicts: dict
        Model methods and hyperparameters to be explored
    n_jobs: int
        number processors to use (Default value = -1)
    _scoring: tuple :
        Scoring metrics used during model training
        (Default value = ("neg_root_mean_squared_error", )
    _quantiles: np.ndarray
        Percentiles at which to report residual error
        (Default value = np.arange(0, 1.01, 0.05)
    _thresholds: np.ndarray
        Range months to use when calcuating residual error
        (Default value = np.arange(0, 49, 3)
    _save_output: bool :
        Whether to save outputs (Default value = False)
    _generate_plots: bool :
        Whether to generate summary plots of model performance
        (Default value = False)
    prefix: str :
        If saving outputs, location to use (Default value = False)

    Returns
    -------
    tuple:
        model_performance_df: pd.DataFrame,
        residError_by_quantile_df: pd.DataFrame,
        quantile_by_residError_df: pd.DataFrame,
        predictions_df: pd.DataFrame,

    """

    model_performance = {}
    residError_by_quantile = []
    quantile_by_residError = []

    # Loop through each model type in the methods dictionin (e.g., Lasso())
    for method_name, method_dict in method_dicts.items():
        best_estimator = train_model(
            method_name=method_name,
            method_dict=method_dict,
            _scoring=_scoring,
            _X=_X,
            _y=_y,
            _groups=_groups,
            n_jobs=n_jobs,
            _save_output=_save_output,
            prefix=prefix,
        )

        m_per, re_quant, quant_re, m_pred = test_model(
            best_estimator=best_estimator,
            method_name=method_name,
            _X_holdout=_X_holdout,
            _y_holdout=_y_holdout,
            _quantiles=_quantiles,
            _thresholds=_thresholds,
            _generate_plots=_generate_plots,
        )

        # Update lists with test results
        model_performance[method_name] = m_per
        residError_by_quantile.append(re_quant)
        quantile_by_residError.append(quant_re)

    model_performance_df = pd.DataFrame.from_dict(model_performance, orient="index")
    residError_by_quantile_df = pd.concat(residError_by_quantile, ignore_index=True)
    quantile_by_residError_df = pd.concat(quantile_by_residError, ignore_index=True)

    # Storing predictions and generating summary plots
    if _save_output:
        m_pred.to_csv(f"{prefix}/{method_name}.predictions.csv")
        model_performance_df.to_csv(f"{prefix}/model_performance.csv")
        residError_by_quantile_df.to_csv(f"{prefix}/model_coverage_by_quantile.csv")
        quantile_by_residError_df.to_csv(f"{prefix}/model_coverage_by_month.csv")

    return (
        model_performance_df,
        residError_by_quantile_df,
        quantile_by_residError_df,
        m_pred,
    )