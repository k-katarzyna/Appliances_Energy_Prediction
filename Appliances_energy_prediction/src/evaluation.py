import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

plt.rcParams["font.size"] = 14.0
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def prediction_benchmarks(data, X_test, y_test):

    n_test = len(y_test)
    X_train_data = data.iloc[:-n_test]
    means = X_train_data.groupby(["day_of_week", "hour", "minute"])["Appliances"].mean()
    historical_means = X_test.apply(lambda row:
                                    means.loc[(row["day_of_week"], row["hour"], row["minute"])], 
                                    axis=1)
    # day_before = enhanced_data.Appliances_24.iloc[-n_test:]
    week_before = data.Appliances_24.shift(1008).iloc[-n_test:]
    
    plt.figure(figsize=(20,6))
    plt.plot(X_test.index, y_test, label="y_test")
    plt.plot(X_test.index, week_before, 
             label=f"week before: MAPE {mean_absolute_percentage_error(y_test, week_before):.2f}, "
                   f"MAE {mean_absolute_error(y_test, week_before):.0f}, "
                   f"RMSE {mean_squared_error(y_test, week_before, squared=False):.0f}")
    plt.plot(X_test.index, historical_means, 
             label=f"historical means: MAPE {mean_absolute_percentage_error(y_test, historical_means):.2f}, "
                   f"MAE {mean_absolute_error(y_test, historical_means):.0f}, "
                   f"RMSE {mean_squared_error(y_test, historical_means, squared=False):.0f}")
    
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%a"))
    plt.legend()
    plt.title("Prediction benchmarks")
    plt.tight_layout()
    plt.show()


def diagnose_errors(y_test, y_pred):

    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    
    ax[0].scatter(y_test, y_pred)
    ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], c="r" )
    ax[0].set_title("y_pred ~ y_test")
    
    errors = y_test - y_pred
    ax[1].scatter(range(len(errors)), errors)
    ax[1].hlines(0, 2000, 0, color="r")
    ax[1].set_title("res ~ i")
    
    ax[2].scatter(y_test, errors)
    ax[2].hlines(0, 2000, 0, color="r")
    ax[2].set_title("res ~ y_test")
    
    ax[3].hist(errors)
    ax[3].set_title("errors")

    plt.tight_layout()
    plt.show()


def evaluate_model(X_train, X_test, y_train, y_test, model):
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    
    plt.figure(figsize=(20,6))
    plt.plot(X_test.index, y_test)
    plt.plot(X_test.index, y_pred,
             color="red",
             label=f"model prediction: MAPE {mape:.3f}, RMSE {mse:.0f}, MAE {mae:.0f}")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%a'))
    plt.legend()
    plt.show()
    print("\n\n")
    diagnose_errors(y_test, y_pred)


def time_series_split(tscv, X_train):
    
    observation_ranges_train = []
    observation_ranges_test = []
    for train_index, test_index in tscv.split(X_train):
        observation_ranges_train.append((train_index.min(), train_index.max() + 1))
        observation_ranges_test.append((test_index.min(), test_index.max() + 1))
    
    plt.figure(figsize=(10, 4))
    bar_width = 0.4
    plt.barh(
        range(len(observation_ranges_train)),
        [end - start for start, end in observation_ranges_train],
        left=[start for start, _ in observation_ranges_train],
        label="Train"
    )
    plt.barh(
        [idx for idx in range(len(observation_ranges_test))],
        [end - start for start, end in observation_ranges_test],
        left=[start for start, _ in observation_ranges_test],
        label="Test"
    )
    plt.xlabel("Observation number", fontdict={"size":12})
    plt.ylabel("Split", fontdict={"size":12})
    plt.title("Observation ranges in TimeSeriesSplit")
    plt.yticks(
        [idx for idx in range(len(observation_ranges_train))],
        [f"Split {i + 1}" for i in range(len(observation_ranges_train))]
    )
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.show()