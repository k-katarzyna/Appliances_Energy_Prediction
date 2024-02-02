import numpy as np
import pandas as pd


@np.vectorize
def scale_annotation(value, factor):
    """
    Converts a numerical value to a string representation scaled down by a given
    factor, with one decimal place.
    """
    return f"{value/factor:.1f}"


def return_train_test_data(data, n_test, xy=False, ohe_drop_first=False):
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
        ohe_drop_first (bool, optional): If True and xy is True, applies one-hot
            encoding to categorical features and drops the first category to avoid
            multicollinearity.

    Returns:
        tuple: Depending on the value of 'xy', it returns either:
               - (pd.DataFrame, pd.DataFrame): train and test sets when xy is False
               or
               - (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series): X_train, X_test,
                   y_train, y_test when xy is True.
    """    
    if xy:
        X = data.drop(["Appliances", "Appliances_24"], axis = 1)
        y = data.Appliances_24

        if ohe_drop_first:
            X = pd.get_dummies(X, dtype=np.uint8, drop_first=True)
        else:
            X = pd.get_dummies(X, dtype=np.uint8)

        X = X.astype(float)
        X_train, y_train = X.iloc[:-n_test], y.iloc[:-n_test]
        X_test, y_test = X.iloc[-n_test:], y.iloc[-n_test:]

        return X_train, X_test, y_train, y_test

    else:
        train, test = data.iloc[:-n_test], data.iloc[-n_test:]
        return train, test


def weighted_mean(row, weights):
    """
    Calculate the weighted mean of values in a row based on the provided weights.

    This function computes the weighted mean for a series of values using specified weights.
    It iterates through each column in the row, multiplies each value by its corresponding
    weight (if a weight is provided), and then divides the total weighted sum by the sum
    of the weights used. The result is rounded to three decimal places.

    Parameters:
        row (pd.Series): A row of values for which the weighted mean is to be calculated
            (a single row from a DataFrame).
        weights (dict): A dictionary where keys correspond to column names in 'row' and values
            are the weights to be applied to each column's value.

    Returns:
        float: The weighted mean of the values in the row, rounded to three decimal places.
    """
    weighted_sum = sum(row[col] * weights[col] for col in row.index if col in weights)
    total_weight = sum(weights[col] for col in row.index if col in weights)
    return weighted_sum / total_weight