import os
from datetime import date
import numpy as np
import pandas as pd
from holidays.countries import Belgium
from holidays.constants import BANK, PUBLIC


class DataEnhancer():

    def __init__(self, data):
        self.data = data.copy()
    
    def _assign_time_of_day(self, hour):   

        return ("morning" if 6 <= hour < 9 else
                "forenoon" if 9 <= hour < 12 else
                "afternoon" if 12 <= hour < 17 else
                "evening" if 17 <= hour < 22 else
                "night")

    def add_datetime_features(self):
        
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

    def add_lagged_features(self, target, lag, how_many, return_new=False):
        
        modified_data = self.data.copy()
        
        lagged_columns = [modified_data[target].shift(i * lag).rename(f"lag_{i}") 
                          for i in range(1, how_many + 1)]
        modified_data = pd.concat([modified_data] + lagged_columns, axis=1)
        modified_data.dropna(inplace=True)
    
        if return_new:
            return modified_data
        else:
            self.data = modified_data
            return self

    def drop_features(self, features):

        self.data.drop(features, axis=1, inplace=True)
        return self

    def mark_high_values(self):

        grouped_by_hour = self.data.groupby(["hour"])["Appliances"]
        grouped_by_hour_and_minute = self.data.groupby(["hour", "minute"])["Appliances"]
        usage = self.data.Appliances

        new_frame = pd.DataFrame({
            "is_high_usage": np.where(usage > np.quantile(usage, 0.9), 1, 0),
            "mean_usage_by_hour": grouped_by_hour.transform("mean"),
            "max_usage_by_hour": grouped_by_hour.transform("max"),
            "max_usage_by_hour_minute": grouped_by_hour_and_minute.transform("max")
        })

        self.data = pd.concat([self.data, new_frame], axis=1)
        return self