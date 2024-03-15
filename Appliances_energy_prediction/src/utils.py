import numpy as np
import pandas as pd
import mlflow


@np.vectorize
def scale_annotation(value, factor):
    """
    Convert a numerical value to a string representation scaled down by a given
    factor, with one decimal place.
    """
    return f"{value/factor:.1f}"


def return_train_test_data(data, n_test, xy=False, ohe=False, drop_first=False):
    """
    Split the dataset into training and testing sets, optionally preparing
    it for machine learning models with OHE.

    This function can either split the data into train and test sets directly
    or it can separate features (X) and target (y), preprocess features with
    one-hot encoding, and then split them into train and test sets.

    Parameters:
        data (pd.DataFrame): The dataset to be split.
        n_test (int): The number of samples to be used in the test set.
        xy (bool, optional): If True, the function separates features and target
            and returns them separately.
        ohe (bool, optional): If True and xy is True, applies one-hot encoding to
            categorical features.
        drop_first (bool, optional): If True (and xy and ohe are True), it drops
            the first category to avoid multicollinearity.

    Returns:
        tuple: Depending on the value of 'xy', it returns either:
               - (pd.DataFrame, pd.DataFrame): train and test sets when xy is False
               or
               - (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series): X_train, X_test,
                   y_train, y_test when xy is True.
    """    
    if xy:
        X = data.drop(["Appliances_24"], axis = 1)
        y = data.Appliances_24
        if ohe:
            if drop_first:
                X = pd.get_dummies(X, dtype=np.uint8, drop_first=True)
            else:
                X = pd.get_dummies(X, dtype=np.uint8)

        X_train, y_train = X.iloc[:-n_test], y.iloc[:-n_test]
        X_test, y_test = X.iloc[-n_test:], y.iloc[-n_test:]

        return X_train, X_test, y_train, y_test

    else:
        train, test = data.iloc[:-n_test], data.iloc[-n_test:]
        return train, test


def model_with_params(model, model_param_grid, preprocess_grid):
    """
    Generate a parameter grid combining model and preprocessing parameters.

    Parameters:
        model: A machine learning model.
        model_param_grid (dict): Grid of hyperparameters for the model.
        preprocess_grid (dict): Grid of hyperparameters for preprocessing.

    Returns:
        tuple: A tuple containing the model and the combined parameter grid.
    """    
    param_grid = {
        **model_param_grid,
        **preprocess_grid
    }
    return model, param_grid


def return_models_with_param_grids(models, models_param_grids,
                                   for_linear_preprocessor, for_tree_preprocessor,
                                   linear_model_names=["ElasticNet", "Ridge"]):
    """
    Generate a list of tuples containing models and corresponding parameter grids.

    Parameters:
        models (list): List of machine learning models.
        models_param_grids (list): List of hyperparameter grids for each model.
        for_linear_preprocessor (dict): Grid of preprocessing params for linear models.
        for_tree_preprocessor (dict): Grid of preprocessing params for tree models.
        linear_model_names (list, optional): Names of linear models.
                                             Default=["ElasticNet", "Ridge"]

    Returns:
        list: A list of tuples, each containing a model and its parameter grid.
    """
    models_with_params = []
    
    for model, grid in zip(models, models_param_grids):
        if model.__class__.__name__ in linear_model_names:
            preprocess_grid = for_linear_preprocessor
        else:
            preprocess_grid = for_tree_preprocessor

        models_with_params.append(model_with_params(model, grid, preprocess_grid))
        
    return models_with_params


def get_metrics_dict(cv_results):
    """
    Extract best scores for multiple metrics from cross-validation results.

    It aggregates the best scores and corresponding scores for RMSE, MAE, MedAE and R2
    metrics, including training scores and additional corresponding scores for the model
    with the best RMSE.

    Parameters:
        cv_results (dict): Results returned by cross-validation.

    Returns:
        dict: A dictionary with keys for each metric and their best and corresponding scores.
    """    
    metrics = {}
    
    for metric in ["rmse", "mae", "medae", "r2"]:
        best_index = np.argmin(cv_results[f"rank_test_{metric}"])

        best_score = cv_results[f"mean_test_{metric}"][best_index]
        best_score_train = cv_results[f"mean_train_{metric}"][best_index]
        metrics[f"best_{metric}"] = np.abs(best_score)
        metrics[f"best_{metric}_train"] = np.abs(best_score_train)

        if metric == "rmse":
            corresponding_mae = cv_results[f"mean_test_mae"][best_index]
            corresponding_medae = cv_results[f"mean_test_medae"][best_index]
            corresponding_r2 = cv_results[f"mean_test_r2"][best_index]
            metrics[f"corresp_mae"] = np.abs(corresponding_mae)
            metrics[f"corresp_medae"] = np.abs(corresponding_medae)
            metrics[f"corresp_r2"] = corresponding_r2

    return metrics


def load_feature_selection_estimators(runs):
    """
    Load best Ridge and ExtraTreesRegressor models based on RMSE for two feature sets.

    Filters MLflow runs for the best performing Ridge and ExtraTreesRegressor models
    trained during 'lags_windows' and 'interactions' experiments. It then loads these
    models from their MLflow artifact URIs.

    Parameters:
        runs (pd.DataFrame): DataFrame containing MLflow run information.

    Returns:
        tuple: Tuple containing four models - two Ridge (linear) models and two
               ExtraTreesRegressor (tree) models, corresponding to the 'lags_windows'
               and 'interactions' experiments, respectively.
    """    
    filtered_runs_l = (runs[(runs["tags.test"] == "lags_windows")
                     & (runs["tags.mlflow.runName"] == "Ridge")])
    best_run_l = filtered_runs_l.sort_values(by="metrics.best_rmse").iloc[0]
    linear_artifact_uri_1 = best_run_l["artifact_uri"]
    linear_model_1 = mlflow.sklearn.load_model(linear_artifact_uri_1)
    
    filtered_runs_t = (runs[(runs["tags.test"] == "lags_windows")
                     & (runs["tags.mlflow.runName"] == "ExtraTreesRegressor")])
    best_run_t = filtered_runs_t.sort_values(by="metrics.best_rmse").iloc[0]
    tree_artifact_uri_1 = best_run_t["artifact_uri"]
    tree_model_1 = mlflow.sklearn.load_model(tree_artifact_uri_1)
    
    filtered_runs_2 = runs[runs["tags.test"] == "interactions"]
    linear_artifact_uri_2 = (filtered_runs_2[filtered_runs_2["tags.mlflow.runName"]
                           .str.contains("Ridge")]["artifact_uri"].iloc[0])
    linear_model_2 = mlflow.sklearn.load_model(linear_artifact_uri)
    tree_artifact_uri_2 = (filtered_runs_2[filtered_runs_2["tags.mlflow.runName"]
                         .str.contains("ExtraTrees")]["artifact_uri"].iloc[0])
    tree_model_2 = mlflow.sklearn.load_model(tree_artifact_uri)

    return linear_model_1, tree_model_1, linear_model_2, tree_model_2


def return_feature_set(artifact_uri):
    """
    Load a model from MLflow artifact and return the set of features selected by RFE.

    Parameters:
        artifact_uri (str): URI to the MLflow artifact where the model is stored.

    Returns:
        list: A list of feature names selected by the RFE process.
    """    
    model = mlflow.sklearn.load_model(artifact_uri)
    feature_set = (model.named_steps["preprocessor"]
                   .get_feature_names_out()
                   [model.named_steps["rfe"].support_])
    feature_set = [str(feature) for feature in feature_set]
    return feature_set