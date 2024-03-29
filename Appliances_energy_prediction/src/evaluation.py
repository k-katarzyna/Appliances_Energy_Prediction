import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LinearRegression

from .utils import return_train_test_data

plt.rcParams["font.size"] = 14.0
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def prediction_benchmarks(data, n_test):
    """
    Generate and plot prediction benchmarks for appliance energy usage forecasting.

    This function divides the provided dataset into training and test sets, then
    creates three sets of predictions:
    1. Data from the week before,
    2. Historical means based on day of the week, hour, and minute,
    3. Predictions from a simple linear regression model.

    The function plots these predictions along with the actual test data for visual
    comparison and calculates root mean squared error (RMSE), mean absolute error (MAE),
    and mean absolute percentage error (MAPE) for each dummy prediction.

    Parameters:
        data (pd.DataFrame): The dataset containing appliance energy usage and other features.
        n_test (int): The number of samples to include in the test set.

    Returns:
        None: The function outputs a matplotlib plot and does not return any value.
    """
    train, test = return_train_test_data(data, n_test)
    
    means = train.groupby(["day_of_week", "hour", "minute"])["Appliances_24"].mean()
    historical_means = test.apply(lambda row:
                                  means.loc[(row["day_of_week"], row["hour"], row["minute"])], 
                                  axis=1)
    
    week_before = data.Appliances_24.shift(144*7).iloc[-n_test:]

    X_train, X_test, y_train, y_test = return_train_test_data(data,
                                                              n_test,
                                                              xy=True,
                                                              ohe=True,
                                                              drop_first=True)
    
    model = TransformedTargetRegressor(make_pipeline(MinMaxScaler(),
                                                     LinearRegression()),
                                       func=np.log,
                                       inverse_func=np.exp)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20,12), sharex=True)
    fig.suptitle("Prediction benchmarks", fontsize=16)

    for ax, title, preds in zip((ax1, ax2, ax3),
                                ("week before",
                                 "historical means",
                                 "linear regression"),
                                (week_before,
                                 historical_means,
                                 y_pred)):
        ax.plot(test.index, y_test, label="y_test", color="tab:blue")
        ax.plot(test.index, preds, label=title, color="red")

        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        medae = median_absolute_error(y_test, preds)
        
        ax.legend([
            "y_test",
            f"{title}: RMSE {rmse:.0f}, MAE {mae:.0f}, MedAE {medae:.0f}"
        ])
        
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%a"))
    plt.tight_layout()
    plt.show()


def time_series_split(tscv, X_train):
    """
    Visualize the training and test splits created by a time series cross-validator.

    This function takes a time series cross-validator object and the training data,
    then generates a visualization of the training and test observation ranges for each
    split. The visualization helps in understanding how the data is partitioned over time.
    
    Parameters:
        tscv (TimeSeriesSplit or similar): A time series cross-validator object.
        X_train (pd.DataFrame or array-like): The training dataset.

    Returns:
        None: This function plots the time series splits and does not return any value.
    """    
    observation_ranges_train = []
    observation_ranges_test = []

    for train_index, test_index in tscv.split(X_train):
        observation_ranges_train.append((train_index.min(), train_index.max() + 1))
        observation_ranges_test.append((test_index.min(), test_index.max() + 1))
    
    plt.figure(figsize=(6, 2.5))
    bar_width = 0.3
    
    palette = sns.color_palette("Paired")
    first_blue = palette[0]
    second_blue = palette[1]
    
    plt.barh(
        range(len(observation_ranges_train)),
        [end - start for start, end in observation_ranges_train],
        left=[start for start, _ in observation_ranges_train],
        label="Train", color=first_blue
    )
    plt.barh(
        [idx for idx in range(len(observation_ranges_test))],
        [end - start for start, end in observation_ranges_test],
        left=[start for start, _ in observation_ranges_test],
        label="Test", color=second_blue
    )
    plt.xlabel("Observation number", fontdict={"size": 12})
    plt.ylabel("Split", fontdict={"size": 12})
    plt.title("Observation ranges in TimeSeriesSplit", fontsize=14)
    plt.yticks(
        [idx for idx in range(len(observation_ranges_train))],
        [f"Split {i + 1}" for i in range(len(observation_ranges_train))]
    )
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.legend()
    plt.show()


def diagnose_errors(y_test, y_pred):
    """
    Visualize the prediction errors in multiple ways to diagnose model performance.

    This function creates four plots to help in understanding the nature of errors
    made by a prediction model:
    1. Scatter plot of predicted values vs actual values.
    2. Scatter plot of error terms against their index.
    3. Scatter plot of error terms against actual values.
    4. Histogram of the error distribution.

    Parameters:
        y_test (array-like): The actual values.
        y_pred (array-like): The predicted values from the model.

    Returns:
        None: The function plots the graphs using matplotlib and does not return any value.
    """
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    
    ax[0].scatter(y_test, y_pred)
    ax[0].plot([y_test.min(), y_test.max()],
               [y_test.min(), y_test.max()],
               c="r" )
    ax[0].set_title("y_pred ~ y_test")
    
    errors = y_test - y_pred
    ax[1].scatter(range(len(errors)), errors)
    ax[1].hlines(0, len(errors), 0, color="r")
    ax[1].set_title("errors ~ i")
    
    ax[2].scatter(y_test, errors)
    ax[2].hlines(0, xmin=min(y_test), xmax=max(y_test), color="r")
    ax[2].set_title("errors ~ y_test")
    
    ax[3].hist(errors)
    ax[3].set_title("errors")

    plt.tight_layout()
    plt.show()


def evaluate_model(X_test, y_test, model, regressor, include_diagnostics=True):
    """
    Evaluate a given model's performance on test data and visualize the results.

    This function uses the provided model to predict values based on X_test, then
    calculates root mean squared error (RMSE), mean absolute error (MAE), median
    absolute error (MedAE) and r2 score (R2). It plots the actual vs predicted values
    and also calls the diagnose_errors function to further analyze the prediction errors.

    Parameters:
        X_test (pd.DataFrame): The test data features.
        y_test (array-like): The actual values for the test data.
        model: Pipeline or fitted model object used for making predictions.
        regressor: The regressor object within `model`, used solely for extracting its
            class name for labeling purposes.
        include_diagnostics (bool, optional): If True, includes diagnostic plots for prediction
            errors. Defaults to True.

    Returns:
        None: This function plots the evaluation results and does not return any value.
    """
    name = regressor.__class__.__name__
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    plt.figure(figsize=(20,5))
    plt.plot(X_test.index,
             y_test,
             label="y_true")
    plt.plot(X_test.index,
             y_pred,
             color="red",
             label=f"prediction, RMSE {rmse:.1f}, MAE {mae:.0f}, MedAE {medae:.0f}, R2 {r2:.2f}")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%a'))
    plt.title(f"Best model's ({name}) performance (on test data)")
    plt.legend()
    plt.show()
    if include_diagnostics:
        print("\n\n")
        diagnose_errors(y_test, y_pred)


def feature_importances(importances, feature_names):
    """
    Create horizontal barplot for feature importances.

    The function helps to assess how the model prioritizes features, which is
    particularly useful in the context of the newly added datetime features
    and random features.

    Parameters:
        importances (np.array): Feature importances values.
        feature_names (list): Feature names for plotting.

    Returns:
        None: This function plots feature importances and does not return any value.
    """
    sorted_indices = importances.argsort()
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_importances = importances[sorted_indices]

    plt.figure(figsize=(6, 13))
    plt.barh(range(len(sorted_names)),
             sorted_importances,
             align="center",
             color="steelblue")
    
    plt.yticks(range(len(sorted_names)), sorted_names, fontsize=10)
    plt.xlabel("Feature Importance", fontdict={"size":12})
    plt.ylabel("Feature Name", fontdict={"size":12})
    plt.title("Feature Importances", fontdict={"size":14})
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.show()