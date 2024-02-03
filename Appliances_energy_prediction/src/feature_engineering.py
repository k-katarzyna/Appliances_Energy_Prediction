import os
from datetime import date
import numpy as np
import pandas as pd
from holidays.countries import Belgium
from holidays.constants import BANK, PUBLIC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest


class DataEnhancer:
    """
    A class used to enhance a dataset with additional features and transformations.

    The class provides methods for adding date and time based features, marking
    high values, adding lagged features, moving averages and sums. It also allows
    dropping specified features and dropping observations with missing values after
    adding lagged/moving features. 

    Attributes:
        data (pd.DataFrame): A DataFrame containing the data to be enhanced.

    Methods:
        add_datetime_features(): Adds date and time related features to the dataset.
        drop_features(features): Drops specified columns from the dataset.
        mark_high_values(): Marks high value records in the dataset.
        add_lagged_features(lags, return_new): Adds lagged features to the dataset.
        add_moving_average(windows, return_new): Adds moving average calculations to
            the dataset.
        add_moving_sum(windows, return_new): Adds moving sum calculations to the dataset.
        dropna(): Drops rows with missing values from the dataset.
    """

    def __init__(self, data):
        """
        Initialize the DataEnhancer with the provided dataset.

        Parameters:
            data (pd.DataFrame): A DataFrame to be enhanced.
        """
        self.data = data.copy()
    
    def _assign_time_of_day(self, hour):   
        """
        Assign a part of the day based on the given hour.

        Parameters:
            hour (int): The hour of the day.

        Returns:
            str: The part of the day corresponding to the hour.
        """
        return ("morning" if 6 <= hour < 9 else
                "forenoon" if 9 <= hour < 12 else
                "afternoon" if 12 <= hour < 17 else
                "evening" if 17 <= hour < 22 else
                "night")
        
    def add_datetime_features(self):
        """
        Add various datetime related features such as day of the week, hour,
        minute, time of day, week of the year, day of the year, holiday status
        and weekend status.

        Returns:
            DataEnhancer: The instance itself for method chaining.
        """        
        belgian_holidays = dict(Belgium(years=2016, categories=(BANK, PUBLIC)))
        dates = self.data.index
        
        new_frame = pd.DataFrame({
            "day_of_week": dates.day_name(),
            "hour": dates.hour,
            "minute": dates.minute,
            "time_of_day": [self._assign_time_of_day(dt.hour) for dt in dates],
            "week_of_year": dates.isocalendar().week,
            "day_of_year": dates.day_of_year,
            "is_holiday": dates.normalize().isin(belgian_holidays.keys()).astype(int),
            "is_weekend": dates.dayofweek.isin([5, 6]).astype(int)
        })  
        self.data = pd.concat([self.data, new_frame], axis=1)
        return self

    def drop_features(self, features):
        """
        Drop specified features from the dataset.

        Parameters:
            features (list of str): List of column names to be dropped.

        Returns:
            DataEnhancer: The instance itself for method chaining.
        """
        self.data.drop(features, axis=1, inplace=True)
        return self

    def mark_high_values(self, quantile_0_1=0.90, quantile_1_2=0.95):
        """
        Mark high value records in the dataset based on predefined criteria.

        Returns:
            DataEnhancer: The instance itself for method chaining.
        """
        usage = self.data.Appliances
        very_high = usage >= np.quantile(usage, quantile_1_2)
        high = (usage < np.quantile(usage, quantile_1_2)) & (usage > np.quantile(usage, quantile_0_1))

        self.data["is_high_usage"] = (np.where(very_high, 2,
                                               np.where(high, 1, 0)))
        return self
        
    def mark_empty_home_days(self, max_usage_threshold=250):
        """
        Mark days when the house is likely empty based on analysis and assumptions
        regarding energy consumption patterns.

        Returns:
            DataEnhancer: The instance itself for method chaining.
        """
        grouped = self.data.groupby("day_of_year")["Appliances"].max()
        empty_home_days = grouped[grouped <= max_usage_threshold].index
        
        self.data["is_empty_home"] = np.where(self.data
                                              .day_of_year
                                              .isin(empty_home_days), 1, 0)
        return self

    def add_lagged_features(self, lags, return_new=False):
        """
        Add lagged features to the dataset.

        Parameters:
            lags (list of int): The list of lag periods.
            return_new (bool): If True, returns a new DataFrame with added features,
                otherwise updates the instance's data.

        Returns:
            DataEnhancer or pd.DataFrame: The instance itself or a new DataFrame
                depending on the 'return_new' parameter.
        """        
        modified_data = self.data.copy()
        
        lagged_columns = [modified_data["Appliances"]
                          .shift(lag)
                          .rename(f"lag_{lag}") 
                          for lag in lags]
        
        modified_data = pd.concat([modified_data] + lagged_columns, axis=1)
    
        if return_new:
            return modified_data.dropna()
        else:
            self.data = modified_data
            return self

    def add_moving_average(self, windows, return_new=False):
        """
        Add moving average calculations to the dataset.

        Parameters:
            windows (list of int): The list of window sizes for moving averages.
            return_new (bool): If True, returns a new DataFrame with added features,
                otherwise updates the instance's data.

        Returns:
            DataEnhancer or pd.DataFrame: The instance itself or a new DataFrame
                depending on the 'return_new' parameter.
        """
        modified_data = self.data.copy()
        
        moving_av_columns = [modified_data["Appliances"]
                             .rolling(window=window_size).mean()
                             .rename(f"moving_av_{window_size}") 
                             for window_size in windows]
        
        modified_data = pd.concat([modified_data] + moving_av_columns, axis=1)
        
        if return_new:
            return modified_data.dropna()
        else:
            self.data = modified_data
            return self

    def add_moving_sum(self, windows, return_new=False):
        """
        Add moving sum calculations to the dataset.

        Parameters:
            windows (list of int): The list of window sizes for moving sums.
            return_new (bool): If True, returns a new DataFrame with added features,
                otherwise updates the instance's data.

        Returns:
            DataEnhancer or pd.DataFrame: The instance itself or a new DataFrame depending
                on the 'return_new' parameter.
        """
        modified_data = self.data.copy()
        
        moving_sum_columns = [modified_data["Appliances"]
                             .rolling(window=window_size).sum()
                             .rename(f"moving_sum_{window_size}") 
                             for window_size in windows]
        
        modified_data = pd.concat([modified_data] + moving_sum_columns, axis=1)
        
        if return_new:
            return modified_data.dropna()
        else:
            self.data = modified_data
            return self

    def dropna(self):
        """
        Drop rows with missing values from the dataset.

        Returns:
            DataEnhancer: The instance itself for method chaining.
        """
        self.data.dropna(inplace=True)
        return self

    def add_cyclic_features(self):
    """
    Add cyclic features to the dataset to capture the cyclical nature of time-related
    variables.

    This method computes and adds sinusoidal and cosinusoidal transformations for hours,
    weekdays and time of day to the dataset. Those could help machine learning models to
    capture the cyclical patterns within time-related data, improving the model's ability
    to make predictions.

    Returns:
        DataEnhancer: The instance itself for method chaining.
    """
        hour = self.data.hour
        self.weekday_ = pd.Series(self.data.index.dayofweek,
                                  index=self.data.index)
        time_of_day_mapping = {"night": 0,
                               "morning": 1,
                               "forenoon": 2,
                               "afternoon": 3,
                               "evening": 4}
        self.time_of_day_ = self.data.time_of_day.map(time_of_day_mapping)
        
        new_frame = pd.DataFrame({
            "hour_sin": hour.apply(
                lambda h: np.sin(h * (2. * np.pi / 24))
            ),
            "hour_cos": hour.apply(
                lambda h: np.cos(h * (2. * np.pi / 24))
            ),
            "weekday_sin": self.weekday_.apply(
                lambda d: np.sin(d * (2. * np.pi / 7))
            ),
            "weekday_cos": self.weekday_.apply(
                lambda d: np.cos(d * (2. * np.pi / 7))
            ),
            "timeofday_sin": self.time_of_day_.apply(
                lambda t: np.sin(t * (2. * np.pi / 5))
            ),
            "timeofday_cos": self.time_of_day_.apply(
                lambda t: np.cos(t * (2. * np.pi / 5))
            )
        })
        self.cyclic_features_ = new_frame.columns.to_list()
        self.data = pd.concat([self.data, new_frame], axis=1)
        return self

    def add_interaction_features(self, datetime=True, climate=False):
    """
    Add interaction features to the dataset based on specified options.

    This method can add interaction features derived from datetime components (hour, minute,
    weekday) and microclimate measurements (temperature and humidity pairs, temperature and
    windspeed). Interaction features are combinations of two or more features that may provide
    additional predictive power to machine learning models.

    Parameters:
        datetime (bool, optional): If True, adds datetime interaction features. Default is True.
        climate (bool, optional): If True, adds climate interaction features. Default is False.

    Returns:
        DataEnhancer: The instance itself for method chaining.
    """
        new_frame = pd.DataFrame()
        
        if datetime:
            datetime_frame = pd.DataFrame({
                "hour_min": self.data.hour + self.data.minute / 60,
                "weekday_hour": (self.weekday_ + 1) * (self.data.hour + 1),
                "weekday_timeofday": (self.weekday_ + 1) * (self.time_of_day_ + 1)
            })
            new_frame = pd.concat([new_frame, datetime_frame], axis=1)

        if climate:
            t_rh_pairs = ([(f"T{i}", f"RH_{i}") for i in range(1, 10) if i != 6]
                          + [("T_out", "RH_out")])
            
            climate_frame = pd.DataFrame({
                **{f"{t}_{rh}" : self.data[t] * self.data[rh] for t, rh in t_rh_pairs},
                "T_out_Windspeed": self.data["T_out"] * self.data["Windspeed"]
            })
            new_frame = pd.concat([new_frame, climate_frame], axis=1)

        self.interaction_features_ = new_frame.columns.to_list()
        self.data = pd.concat([self.data, new_frame], axis=1)
        return self


class AnomaliesMarker(BaseEstimator, TransformerMixin):
    """
    A class used for marking anomalies in a dataset using the Isolation Forest algorithm.

    This class is a custom transformer that extends the functionality of scikit-learn's 
    BaseEstimator and TransformerMixin. It is designed to fit an Isolation Forest model 
    on the data for anomaly detection and subsequently mark the anomalies in the dataset.

    Attributes:
        model (IsolationForest): The Isolation Forest model used for anomaly detection.

    Methods:
        fit(X, y=None): Fits the Isolation Forest model on the dataset.
        transform(X): Transforms the dataset by marking the anomalies.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the AnomaliesMarker with an Isolation Forest model.

        Parameters:
            **kwargs: Arbitrary keyword arguments that are passed to the IsolationForest
                constructor.
        """        
        self.model = IsolationForest(n_jobs=-1, random_state=42, **kwargs)

    def fit(self, X, y=None):
        """
        Fit the Isolation Forest model to the dataset.

        Parameters:
            X (pd.DataFrame): The input samples.
            y (ignored): Not used, present here for API consistency by convention.

        Returns:
            AnomaliesMarker: The instance itself.
        """        
        self.model.fit(X)
        return self

    def transform(self, X):
        """
        Apply the fitted Isolation Forest model to the dataset and mark anomalies.

        Anomalies are marked in a new column named 'anomalies', with 1 indicating
        an anomaly and 0 indicating normal.

        Parameters:
            X (pd.DataFrame): The input samples.

        Returns:
            pd.DataFrame: The transformed DataFrame with an additional 'anomalies' column.
        """        
        X_tr = X.copy()
        
        preds = self.model.predict(X_tr)
        X_tr["anomalies"] = np.where(preds == -1, 1, 0)
        
        return X_tr