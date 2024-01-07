import os
from datetime import date
from holidays.countries import Belgium
from holidays.constants import BANK, PUBLIC
import pandas as pd


# def holidays():
#     holidays_path = os.path.join("data", "belgian_holidays", "belgian_holidays_2016.csv")
#     holidays = pd.read_csv(holidays_path)
#     filtered_holidays = holidays[holidays['Holiday Type'].isin(['National holiday', 'Bank holiday'])]
#     holiday_dates = filtered_holidays['Date'].to_list()
    
#     return pd.to_datetime(holiday_dates).date

def assign_time_of_day(hour):
    
    if 23 <= hour or hour < 5:
        return "night"
    elif 5 <= hour < 8:
        return "morning"
    elif 8 <= hour < 12:
        return "forenoon"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 23:
        return "evening"
    
    
def add_datetime_features(data):
    
    # belgian_holidays = holidays()
    belgian_holidays = dict(Belgium(years=2016, categories=(BANK, PUBLIC)))
    
    new_frame = pd.DataFrame(
        {
            "day_of_week": data.index.day_name(),
            "hour": data.index.hour,
            "time_of_day": [assign_time_of_day(dt.hour) for dt in data.index],
            "week_of_year": data.index.isocalendar().week,
            "day_of_year": data.index.day_of_year,
            "is_holiday": data.index.normalize().isin(belgian_holidays.keys()).astype(int),
            "is_weekend": data.index.dayofweek.isin([5, 6]).astype(int)
        }
    )
    return pd.concat([data, new_frame], axis=1)


def add_lagged_features(data, lag, how_many):
    
    data_t = data.copy()
    
    lagged_columns = [data_t.Appliances.shift(i*lag).rename(f"lag_{i}") for i in range(1, how_many+1)]
    data_t = pd.concat([data_t] + lagged_columns, axis=1)
    data_t.dropna(inplace=True)

    return data_t