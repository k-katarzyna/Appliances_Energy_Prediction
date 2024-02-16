import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import RFE

from .utils import get_metrics_dict, return_train_test_data

SEED = 42
SEARCH_CV_PARAMS = {
    "scoring": {"rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error",
                "medae": "neg_median_absolute_error",
                "r2": "r2"},
    "refit": "rmse",
    "n_jobs": -1,
    "return_train_score": True,
    "error_score": "raise"
}


def optimize_params(preprocessor, model, param, X_train, y_train, tss,
                    n_iter=100, fit_kw={}):
    """
    Optimize model parameters using RandomizedSearchCV with specified preprocessor
    and cross-validation strategy.

    Creates a pipeline comprising a preprocessor and a model. Then optimizes the model's
    parameters. Utilizes RandomizedSearchCV for optimization, based on multiple scoring
    metrics and allows for refitting on RMSE. Additional fitting options can be passed.

    Parameters:
        preprocessor (Transformer): Preprocessing steps applied to data before modeling.
        model (Estimator): The machine learning model to optimize.
        param (dict): Parameter distributions for sampling during optimization.
        X_train, y_train: Training data (features and target).
        tss (TimeSeriesSplit): Strategy for time series cross-validation.
        n_iter (int, optional): Number of parameter settings to sample (default 100).
        fit_kw (dict, optional): Additional kwargs for the fit method.

    Returns:
        tuple: Optimized model and a dictionary of metrics for best parameters.

    Note:
        - 'SEED' and 'SEARCH_CV_PARAMS' are used for configuration.
    """  
    pipeline = Pipeline([("preprocessor", preprocessor),
                         ("model", model)])

    optimizer = RandomizedSearchCV(estimator=TransformedTargetRegressor(pipeline), 
                                   param_distributions=param,
                                   cv=tss,
                                   n_iter=n_iter,
                                   random_state=SEED,
                                   **SEARCH_CV_PARAMS)

    optimizer.fit(X_train, y_train, **fit_kw)
    metrics = get_metrics_dict(optimizer.cv_results_)

    return optimizer, metrics


def grid_search(estimator, params, X_train, y_train,
                tss, fit_kw={}):
    """
    Perform grid search over specified parameter grid with a given estimator and
    cross-validation strategy.

    Parameters:
        estimator: The model or pipeline to optimize.
        params (dict): Grid of parameters to search over.
        X_train, y_train: Training data (features and target).
        tss (TimeSeriesSplit): Strategy for time series cross-validation.
        fit_kw (dict, optional): Additional kwargs for the fit method.

    Returns:
        tuple: Optimized model and a dictionary of metrics for the best parameters.
    """    
    optimizer = GridSearchCV(estimator, 
                             params,
                             cv=tss,
                             **SEARCH_CV_PARAMS)

    optimizer.fit(X_train, y_train, **fit_kw)
    metrics = get_metrics_dict(optimizer.cv_results_)

    return optimizer, metrics


def pipeline_recomposition_with_rfe(model, name, rfe_estimators, common_preprocessor=True):
    """
    Reassemble a pipeline including RFE based on the model type and chosen estimators.

    Constructs a new pipeline using Recursive Feature Elimination (RFE) with a specified
    model and preprocessor. It configures RFE and the final model based on the model type.

    Parameters:
        model: The model to be used in the final step of the pipeline.
        name (str): Model name to determine specific configurations.
        rfe_estimators (list): List containing estimators to configure RFE, the first for
            linear models (Ridge, ElasticNet) and the second one for tree based models.
        common_preprocessor (bool, optional): Flag to use the common preprocessor or
            model-specific. Defaults to True.

    Returns:
        tuple: The reassembled pipeline, training data subset used for RFE, and feature
            importance attribute name.
    """    
    if name in ["Ridge", "ElasticNet"]:
        rfe_estimator = rfe_estimators[0]
        importance = "coef_"
    else:
        rfe_estimator = rfe_estimators[1]
        importance = "feature_importances_"

    rfe_m = rfe_estimator.regressor_.named_steps["model"]
    rfe_ttt = rfe_estimator.transformer_
    rfe_ttr_estimator = TransformedTargetRegressor(regressor=rfe_m,
                                                   transformer=rfe_ttt)

    if common_preprocessor:
        preprocessor = rfe_estimator.regressor_.named_steps["preprocessor"]
    else:
        preprocessor = model.regressor_.named_steps["preprocessor"]
        
    target_transformer = model.transformer_
    tt_regressor = model.regressor_.named_steps["model"]
    last_step_estimator = TransformedTargetRegressor(regressor=tt_regressor,
                                                     transformer=target_transformer)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("rfe", RFE(rfe_ttr_estimator)),
        ("model", last_step_estimator)
    ])

    X_train, _ = return_train_test_data(enhanced_data[rfe_estimator.regressor_.feature_names_in_],
                                        n_test)

    return pipeline, X_train, importance