from itertools import product

import numpy as np
import pandas as pd
from holidays.countries import Belgium
from holidays.constants import BANK, PUBLIC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import FunctionTransformer


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
        mark_high_values(quantile_0_1, quantile_1_2): Marks high value records in
            the dataset.
        mark_empty_home_days(max_usage_threshold): Marks days with minimal energy
            consumption.
        add_lagged_features(lags, return_new): Adds lagged features to the dataset.
        add_moving_average(windows, return_new): Adds moving average calculations to
            the dataset.
        add_moving_sum(windows, return_new): Adds moving sum calculations to the dataset.
        add_cyclic_features(): Adds cyclic features for hours, weekdays and time of day.
        add_interaction_features(datetime, climate): Adds datetime and microclimate
            interaction features.
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
        
        Parameters:
            quantile_0_1 (float): The lower quantile threshold to classify records
                as high usage. Default is 0.90. Records above this threshold but below
                the `quantile_1_2` are marked as high usage (1).
            quantile_1_2 (float): The upper quantile threshold to classify records as very
                high usage. Default is 0.95. Records above this threshold are marked as very
                high usage (2).

        Returns:
            DataEnhancer: The instance itself for method chaining.
        """
        usage = self.data.Appliances
        very_high = usage >= np.quantile(usage, quantile_1_2)
        high = ((usage < np.quantile(usage, quantile_1_2))
                & (usage > np.quantile(usage, quantile_0_1)))

        self.data["is_high_usage"] = (np.where(very_high, 2,
                                               np.where(high, 1, 0)))
        return self
        
    def mark_empty_house_days(self, max_usage_threshold=250):
        """
        Mark days when the house is likely empty based on analysis and assumptions
        regarding energy consumption patterns.

        Parameters:
            max_usage_threshold (int): The maximum energy usage threshold (in Wh)
                for considering a home empty for the day. Default is 250 Wh.

        Returns:
            DataEnhancer: The instance itself for method chaining.
        """
        grouped = self.data.groupby("day_of_year")["Appliances"].max()
        empty_home_days = grouped[grouped <= max_usage_threshold].index
        
        self.data["is_empty_house"] = np.where(self.data
                                               .day_of_year
                                               .isin(empty_home_days), 1, 0)
        return self

    def add_lagged_features(self, features, lags, return_new=False):
        """
        Add lagged features to the dataset.

        Parameters:
            features (list of strings): The list of feature names.
            lags (list of int): The list of lag periods.
            return_new (bool): If True, returns a new DataFrame with added features,
                otherwise updates the instance's data.

        Returns:
            DataEnhancer or pd.DataFrame: The instance itself or a new DataFrame
                depending on the 'return_new' parameter.
        """        
        modified_data = self.data.copy()
        
        lagged_columns = [modified_data[feature]
                          .shift(lag)
                          .rename(f"lag_{feature}_{lag}") 
                          for feature, lag in product(features, lags)]

        modified_data = pd.concat([modified_data] + lagged_columns, axis=1)
    
        if return_new:
            return modified_data.dropna()
        else:
            self.data = modified_data
            return self

    def add_moving_average(self, features, windows, return_new=False):
        """
        Add moving average calculations to the dataset.

        Parameters:
            features (list of strings): The list of feature names.
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
                             .rename(f"mov_av_{feature}_{window_size}") 
                             for feature, window_size in product(features, windows)]
        
        modified_data = pd.concat([modified_data] + moving_av_columns, axis=1)
        
        if return_new:
            return modified_data.dropna()
        else:
            self.data = modified_data
            return self

    def add_moving_sum(self, features, windows, return_new=False):
        """
        Add moving sum calculations to the dataset.

        Parameters:
            features (list of strings): The list of feature names.
            windows (list of int): The list of window sizes for moving sums.
            return_new (bool): If True, returns a new DataFrame with added features,
                otherwise updates the instance's data.

        Returns:
            DataEnhancer or pd.DataFrame: The instance itself or a new DataFrame
            depending on the 'return_new' parameter.
        """
        modified_data = self.data.copy()
        
        moving_sum_columns = [modified_data["Appliances"]
                             .rolling(window=window_size).sum()
                             .rename(f"mov_sum_{feature}_{window_size}") 
                             for feature, window_size in product(features, windows)]
        
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

    def add_interaction_features(self, datetime=True, climate=False):
        """
        Add interaction features to the dataset based on specified options.
    
        This method can add interaction features derived from datetime components (hour,
        minute, weekday) and microclimate measurements (temperature and humidity pairs,
        temperature and windspeed).
    
        Parameters:
            datetime (bool, optional): If True, adds datetime interaction features. Default
                is True.
            climate (bool, optional): If True, adds climate interaction features. Default
                is False.
    
        Returns:
            DataEnhancer: The instance itself for method chaining.
        """
        new_frame = pd.DataFrame()
        time_of_day_mapping = {"night": 0,
                               "morning": 1,
                               "forenoon": 2,
                               "afternoon": 3,
                               "evening": 4}
        time_of_day = self.data.time_of_day.map(time_of_day_mapping).astype(int)
        weekday = pd.Series(self.data.index.dayofweek,
                            index=self.data.index)
        
        if datetime:
            datetime_frame = pd.DataFrame({
                "hour_min": self.data.hour + self.data.minute / 60,
                "weekday_hour": (weekday + 1) * (self.data.hour + 1),
                "weekday_timeofday": (weekday + 1) * (time_of_day + 1)
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
        self.input_features = None

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
        self.input_features = X.columns.tolist()
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

    def get_feature_names_out(self, input_features):
        """
        Get the names of the output features.
        """
        return np.array(self.input_features + ["anomalies"])


def cyclical_feature_encoder(order_dict=None, n_values=None):
    """
    Create a function to encode a feature into its cyclical representation using
    sine and cosine transformation.

    This function is designed for use with FunctionTransformer. It generates sine
    and cosine features from a given feature to capture its cyclical nature, such
    as hour or days of the week. It supports encoding based on a specified order
    for categorical features (`order_dict`) or directly for numerical features using
    the total number of unique values (`n_values`).

    Parameters:
        order_dict (dict, optional): Maps categorical feature values to integers indicating
            their order.
        n_values (int, optional): Specifies the cycle length for numerical features.

    Returns:
        function: Encodes a feature into sine and cosine components.

    Raises:
        ValueError: If both `order_dict` and `n_values` are None.

    Example:
        # For categorical features
        encoder = cyclical_feature_encoder(order_dict={'Mon': 0, 'Tue': 1, ...})
        transformer = FunctionTransformer(func=encoder)
        
        # For numerical features
        encoder = cyclical_feature_encoder(n_values=24)
        transformer = FunctionTransformer(func=encoder)
    """  
    if order_dict:
        n_values = len(order_dict)
        transform_func = lambda x: np.vectorize(order_dict.get)(x)
    elif n_values is None:
        raise ValueError("'order_dict' (for categorical features) "
                         "or 'n_values' (for numerical features) must be provided.")
    else:
        transform_func = lambda x: x

    def encode_feature(feature):
        """
        Encode a feature into its cyclical representation using sine and cosine
        transformation.
        
        This inner function applies the encoding logic.

        Parameters:
            feature (array-like): The feature to encode.

        Returns:
            np.ndarray: A two-dimensional array with sine and cosine encoded
                cyclical features.
        """
        encoded = transform_func(feature)
        
        encoded_sin = np.sin(2. * np.pi * encoded / n_values)
        encoded_cos = np.cos(2. * np.pi * encoded / n_values)
        
        return np.column_stack((encoded_sin, encoded_cos))
    
    return encode_feature


def generate_output_feature_names(transformer, input_features):
    """
    Generate names for cyclical features ('_sin', '_cos') for FunctionTransformer.

    Parameters:
        transformer: The FunctionTransformer (unused, for compatibility).
        input_features (list): Original feature names.

    Returns:
        list: Names with '_sin' and '_cos' suffixes for cyclical features.

    Example:
        transformer = FunctionTransformer(func=cyclical_feature_encoder(...),
                                    feature_names_out=generate_output_feature_names)
    """
    return [f"{feature}_{suffix}"
            for feature in input_features
            for suffix in ['sin', 'cos']]


def return_function_transformers():
    """
    Create and return cyclical and target transformers for feature engineering.

    Returns:
        tuple: Cyclical encoders for day of week, time of day and hour, along with
               log and square root transformers for target variable transformation.
    """
    # cyclical encoding for weekdays
    day_of_week_order = {"Monday": 0, "Tuesday": 1, "Wednesday": 2,
                         "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    encode_day_of_week = cyclical_feature_encoder(order_dict=day_of_week_order)
    day_encoder = FunctionTransformer(func=encode_day_of_week,
                                      feature_names_out=generate_output_feature_names)
    
    # cyclical encoding for time of day
    time_of_day_order = {"night": 0, "morning": 1, "forenoon": 2,
                         "afternoon": 3, "evening": 4}
    encode_time_of_day = cyclical_feature_encoder(order_dict=time_of_day_order)
    time_encoder = FunctionTransformer(func=encode_time_of_day,
                                       feature_names_out=generate_output_feature_names)

    # cyclical encoding for hour
    encode_hour = cyclical_feature_encoder(n_values=24)
    hour_encoder = FunctionTransformer(func=encode_hour,
                                       feature_names_out=generate_output_feature_names)

    # target transformers
    log_transformer = FunctionTransformer(func=np.log, inverse_func=np.exp)
    sqrt_transformer = FunctionTransformer(func=np.sqrt, inverse_func=np.square)

    return day_encoder, time_encoder, hour_encoder, log_transformer, sqrt_transformer


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    A transformer for selecting specific columns from a DataFrame.

    This class is useful in a preprocessing pipeline to select a subset of data
    for further analysis or transformation.

    Parameters:
        columns (list of str): The names of the columns to select from the DataFrame.

    Methods:
        fit(X, y=None): Does nothing, only for compatibility with the scikit-learn
            transformer interface.
        transform(X): Returns the specified columns from X.
        get_feature_names_out(): Returns selected feature names.
    """

    def __init__(self, columns=list):
        """
        Initialize the ColumnSelector with the names of the columns to select.
        """
        self.columns = columns
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data. This method doesn't do anything as column
        selection does not require fitting.

        Parameters:
            X (pd.DataFrame): Data to fit.
            y (ignored): Not used, present here for API consistency by convention.

        Returns:
            self: The instance itself.
        """
        return self
    
    def transform(self, X):
        """
        Transform the data by selecting the specified columns.

        Parameters:
            X (pd.DataFrame): The input data to transform.

        Returns:
            DataFrame: A DataFrame containing only the specified columns from the input data.
        """
        return X[self.columns]

    def get_feature_names_out(self, input_features=None):
        """
        Get the names of the output features.
    
        This method returns the names of the features that have been selected for processing.
        It is useful for understanding which features are being passed through the pipeline.
    
        Returns:
            np.ndarray: An array of selected feature names.
        """
        return np.array(self.columns)