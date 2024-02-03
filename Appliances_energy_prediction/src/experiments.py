from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

SEED = 42


def optimize_params(preprocessor, model, param, X_train, y_train, tss,
                    n_iter=100, rscv_kw={}, fit_kw={}):
    """
    Optimize parameters for a model within a pipeline using RandomizedSearchCV.

    This function creates a pipeline with a given preprocessor and model, and then
    uses RandomizedSearchCV to optimize the model's parameters based on specified
    scoring metrics. It supports multi-metric evaluation (RMSE, MAE, MAPE), refitting
    with RMSE and passing additional keyword arguments to the search and fit methods.

    Parameters:
        preprocessor (Transformer): The preprocessing step(s) to be applied before the model.
        model (Estimator): The model to be optimized.
        param (dict): The parameter distribution to sample during optimization.
        X_train (pd.DataFrame or np.ndarray): Training data features.
        y_train (pd.Series or np.ndarray): Training data target.
        tss (TimeSeriesSplit): Cross-validation strategy for time series data.
        n_iter (int, optional): Number of parameter settings sampled. Default is 100.
        rscv_kw (dict, optional): Additional keyword arguments for RandomizedSearchCV.
        fit_kw (dict, optional): Additional keyword arguments for the fit method.

    Returns:
        RandomizedSearchCV: The RandomizedSearchCV instance after fitting.

    Note:
        - SEED should be defined externally.
    """   
    pipeline = Pipeline([("preprocessor", preprocessor),
                         ("model", model)])

    optimizer = RandomizedSearchCV(estimator=TransformedTargetRegressor(pipeline), 
                                   param_distributions=param, 
                                   cv=tss,
                                   n_iter=n_iter, 
                                   scoring={"rmse": "neg_root_mean_squared_error",
                                            "mae": "neg_mean_absolute_error",
                                            "mape": "neg_mean_absolute_percentage_error"},
                                   refit="rmse",
                                   n_jobs=-1,
                                   error_score="raise",
                                   random_state=SEED,
                                   **rscv_kw)

    optimizer.fit(X_train, y_train, **fit_kw)

    return optimizer