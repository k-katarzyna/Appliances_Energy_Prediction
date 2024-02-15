from itertools import product

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from .utils import scale_annotation

plt.rcParams["font.size"] = 14.0
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def histplots_grid(n_rows, n_cols, data, features=None):
    """
    Generate a grid of histograms for specified features in the given dataset.

    This function creates a grid of histograms, each corresponding to a different 
    feature in the dataset. It allows for a quick and comprehensive overview of
    the distribution of multiple features. The number of rows and columns for
    the grid layout can be specified and, optionally, a subset of features to be
    plotted. If no features are specified, histograms for all continuous columns
    are generated.

    Parameters:
        n_rows (int): The number of rows in the grid.
        n_cols (int): The number of columns in the grid.
        data (pd.DataFrame): The dataset containing the features.
        features (list, optional): A list of feature names (column names) to be plotted.
            If None, all continuous features are plotted.

    Returns:
        None: This function plots the histograms and does not return any value.
    """
    if features is None:
        num_unique_values = data.select_dtypes(exclude="object").nunique()
        features = [col for col in data.select_dtypes([int, float]).columns
                    if num_unique_values[col] > 2]
    
    width = n_cols * 3.2
    height = n_rows * 2.4
    
    fix, axes = plt.subplots(n_rows, n_cols, figsize=(width, height), squeeze=False)
    
    for i, feature in enumerate(features):
        row, col = divmod(i, n_cols)
        
        if row >= n_rows:
            break
        
        ax = axes[row, col]
        sns.histplot(data[feature], ax=ax)
        ax.set_title(feature)
        ax.set_xlabel(None)
        ax.set_ylabel(None) 

    for j in range(i + 1, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axes[row, col].axis("off")

    plt.tight_layout()    
    plt.show()


def energy_consumption_all_time(appliances):
    """
    Visualize daily energy consumption of appliances over time.

    This function creates two heatmaps:
    1. The first heatmap displays the total daily appliance energy usage
    for each day of the month across different months.
    2. The second heatmap shows the average daily energy usage by month.

    The function is designed to provide insights into daily and monthly usage
    patterns of appliances. It uses resampling to calculate daily averages and
    groups data to generate visualizations.

    Parameters:
        appliances (pd.Series or pd.DataFrame): A time-series data representing
            appliance energy consumption.

    Returns:
        None: This function plots the heatmaps and does not return any value.
    """
    heatmap_data = (appliances.groupby([appliances.index.day,
                                        appliances.index.month])
                    .sum()
                    .unstack(level=0))
    
    monthly_means = heatmap_data.mean(axis=1).to_frame()

    fig, (ax1, ax2) = plt.subplots(1, 2,
                                   figsize=(20, 4),
                                   gridspec_kw={"width_ratios": [30, 1]})
    vmin = heatmap_data.min().min()
    vmax = heatmap_data.max().max()
    months = ["Jan", "Feb", "Mar", "Apr", "May"]

    sns.heatmap(heatmap_data,
                cmap="Blues",
                ax=ax1,
                vmin=vmin,
                vmax=vmax,
                cbar_kws={"label": "[Wh]"})
    
    ax1.set_title("Daily Appliances Usage", fontsize=18)
    ax1.set_xlabel("Day of month")
    ax1.set_ylabel("Month")
    ax1.set_yticklabels(months, rotation=0)
    ax1.set_xticks(np.arange(31) + .5)
    ax1.set_xticklabels(range(1, 32))

    sns.heatmap(monthly_means,
                ax=ax2,
                vmin=vmin,
                vmax=vmax,
                cmap="Blues",
                cbar=False,
                annot=scale_annotation(monthly_means.values, 1000),
                fmt="s")
    
    ax2.set_title("Daily means [kWh]")
    ax2.set_ylabel(None)
    ax2.set_yticklabels(months, rotation=0)
    ax2.set_xticks([])

    plt.tight_layout()
    plt.show()
    
    
def appliances_vs_lights_plot(appliances, lights):
    """
    Plot energy consumption of appliances alongside light energy consumption
    over time.

    This function creates a dual-axis line plot where one axis represents the energy
    consumption of appliances and the other represents the energy consumption of lights.
    This visualization is useful for comparing two related time-series data sets
    and understanding their relationship over time.

    Parameters:
        appliances (pd.Series): Time-series data representing the energy consumption
            of appliances.
        lights (pd.Series): Time-series data representing the energy consumption of lights.

    Returns:
        None: This function plots the dual-axis line plot and does not return any value.
    """    
    fig, ax1 = plt.subplots(1, 1, figsize=(20, 4))
    ax1.set_xlabel("Date")

    x_labels = appliances.index.strftime("%Y-%m-%d")
    x_labels = x_labels[::int(len(x_labels) / 6)]

    sns.lineplot(x=appliances.index,
                 y=appliances,
                 ax=ax1,
                 color="midnightblue",
                 linestyle="-",
                 label="Appliances energy consumption")
    
    ax1.set_ylabel("Appliances [Wh]")
    ax1.set_title("Appliances energy consumptions vs. lights energy consumption")
    ax1.set_xticks(x_labels)
    ax1.set_xticklabels(x_labels)

    ax2 = ax1.twinx()
    sns.lineplot(x=lights.index,
                 y=lights,
                 ax=ax2,
                 color="skyblue",
                 linestyle="-",
                 label="Light energy consumption",
                 alpha=0.7)
    ax2.set_ylabel("Lights [Wh]")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="best")
    plt.show()


def time_series_decomposition(result_day, result_week, name):
    """
    Visualize the time series decomposition into trend, seasonality, and residuals.

    This function plots the trend, seasonality, and residuals of a time series data.
    It uses the results from two different decompositions - one at a daily frequency
    and another at a weekly frequency. The function creates four subplots: daily trend,
    weekly trend, weekly seasonality, and weekly residuals.

    Parameters:
        result_day (statsmodels.tsa.seasonal.DecomposeResult): The decomposition result
            at a daily frequency.
        result_week (statsmodels.tsa.seasonal.DecomposeResult): The decomposition result
            at a weekly frequency.
        name (str): Name of column to use in chart suptitle.

    Returns:
        None: This function plots the decomposition results and does not return any value.
    """
    fig, axes = plt.subplots(4, 1, figsize=(20, 12), sharex=True)
    fig.suptitle(f"{name} decomposition", fontsize=20)
    
    sns.lineplot(data=result_day.trend, ax=axes[0])
    axes[0].legend(["Trend (day)"])
    
    sns.lineplot(data=result_week.trend, ax=axes[1])
    axes[1].legend(["Trend (week)"])
    
    sns.lineplot(data=result_week.seasonal, ax=axes[2])
    axes[2].legend(["Seasonality (week)"])
    
    sns.scatterplot(data=result_week.resid, ax=axes[3])
    axes[3].legend(["Residual (week)"])
    
    plt.tight_layout()
    plt.show()


def autocorrelation_plot(time_series, lags):
    """
    Plot the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)
    of a time series.

    This function is used to analyze the autocorrelation and partial autocorrelation
    of a time series data. It helps in identifying the lagged relationships and is useful
    in time series forecasting model selection.

    Parameters:
        time_series (pd.Series): The time series data for which ACF and PACF are to be plotted.
        lags (int): The number of lags to be included in the ACF and PACF plots.

    Returns:
        None: This function plots the ACF and PACF and does not return any value.
    """    
    plt.figure(figsize=(20, 8))
    plt.subplot(221)
    plot_acf(time_series, ax=plt.gca(), lags=lags)
    plt.title(f"ACF for {time_series.name}")
    plt.subplot(222)
    plot_pacf(time_series, ax=plt.gca(), lags=lags)
    plt.title(f"PACF for {time_series.name}")
    plt.tight_layout()
    plt.show()
    
    
def consumption_by_day_and_hour(data):
    """
    Visualize the average energy consumption of appliances by day of the week and hour.

    This function creates a heatmap representation of the average energy usage of appliances,
    segmented by each hour of the day and each day of the week. Additionally, it includes
    heatmaps to show the daily and hourly average energy consumption. This visualization helps
    in identifying patterns and trends in energy usage over different times of the day and days
    of the week.

    Parameters:
        data (pd.DataFrame): The dataset containing the 'Appliances' energy usage along with
            'day_of_week' and 'hour' columns.

    Returns:
        None: This function plots the heatmaps and does not return any value.
    """
    grouped_data = (data
                    .groupby(["week_of_year", "day_of_week", "hour"])["Appliances"]
                    .sum()
                    .unstack(level=2)
                    .groupby("day_of_week")
                    .mean())
    
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    grouped_data = grouped_data.reindex(order)

    daily_means = grouped_data.sum(axis=1).to_frame()
    hourly_means = grouped_data.mean(axis=0).to_frame().T
    
    fig = plt.figure(figsize=(20, 6.5), constrained_layout=True)
    
    gs = fig.add_gridspec(2, 2,
                          width_ratios=[26, 1],
                          height_ratios=[6, 0.8], 
                          left=0.1,
                          right=0.9,
                          bottom=0.1,
                          top=0.9,
                          wspace=0.05,
                          hspace=0.05)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    
    vmin = grouped_data.min().min()
    vmax = grouped_data.max().max()
    
    sns.heatmap(grouped_data,
                cmap="Blues",
                ax=ax1,
                vmin=vmin,
                vmax=vmax,
                cbar=True,
                cbar_kws={"label": "[Wh]"})
    
    ax1.set_title("Hourly average appliances usage by weekday", fontsize=18)
    ax1.set_yticklabels([day[:3] for day in order])
    ax1.set_xticklabels(range(0,24))
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Day of Week")
    
    sns.heatmap(daily_means,
                ax=ax2,
                cmap=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True),
                cbar=False,
                annot=scale_annotation(daily_means.values, 1000),
                fmt="s")
    
    ax2.set_title("Daily means [kWh]")
    ax2.set_xticks([])
    ax2.set_yticklabels([day[:3] for day in order])
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    
    sns.heatmap(hourly_means,
                ax=ax3,
                vmin=vmin,
                vmax=vmax,
                cmap="Blues",
                cbar=False,
                annot=False)
    
    ax3.set_title("Hourly Means", fontsize=16)
    ax3.xaxis.tick_top()
    ax3.set_xticklabels("")
    ax3.set_xlabel("")
    ax3.set_yticks([])
    plt.show()


class WeeklyDataVisualizer:
    """
    A class for visualizing weekly data in various formats including line plots and heatmaps.

    This class is designed to provide flexible and informative visualizations for time series
    data on a weekly basis. It supports plotting individual weeks, multiple weeks and generating
    heatmaps to analyze patterns across different days and hours.

    Attributes:
        data (pd.DataFrame): The dataset containing time series data along with 'week_of_year',
            'day_of_week', and 'hour' columns.

    Methods:
        plot_one_week(week, columns, **kwargs): Plots line graphs for specified columns in
            a given week.
        plot_many_weeks(weeks, columns, single_plot=True, **kwargs): Plots line graphs for
            specified columns across multiple weeks.
        plot_heatmap(weeks, col, **kwargs): Generates heatmaps for a specified column across
            multiple weeks.
    """
    
    def __init__(self, data):
        """
        Initialize the WeeklyDataVisualizer with the provided dataset.

        Parameters:
            data (pd.DataFrame): A DataFrame containing the data for visualization.
        """        
        self.data = data

    def _get_week_data(self, week, col):
        """
        Retrieve data for a specific week and column.

        Parameters:
            week (int): The week number for which data is required.
            col (str): The column name for which data is required.

        Returns:
            pd.DataFrame or None: The subset of data for the specified week and column
                or None if no data is available.
        """        
        week_data = self.data[self.data["week_of_year"] == week]
        if not week_data.empty and col in week_data.columns:
            return week_data
        return None

    def _plot_week_data(self, ax, week, col, label, sharex_using_weekdays=False, **kwargs):
        """
        Plot the data of a specific week on the provided axes.

        Parameters:
            ax (matplotlib.axes.Axes): The axes on which to plot the data.
            week_data (pd.DataFrame): The data for the specific week.
            week (int): The week number.
            col (str): The column name of the data to be plotted.
            label (str): The label for the plot.
            sharex_using_weekdays (bool): If True, use weekdays as the shared x-axis.
            **kwargs: Additional keyword arguments for the seaborn lineplot.
        """
        week_data = self._get_week_data(week, col)
        if week_data is not None:
            
            day_offset = week_data.index[0].weekday()
            time_offset = (day_offset * 24 * 6
                           + week_data.index[0].hour * 6
                           + week_data.index[0].minute // 10)
            
            if sharex_using_weekdays:
                start_date = "2016-01-10"
            else:
                start_date = week_data.index[0].date()
                
            full_week_x_axis = pd.date_range(start=start_date,
                                             periods=7*24*6,
                                             freq="10T")
            adjusted_index = full_week_x_axis[time_offset:time_offset + len(week_data)]
            
            sns.lineplot(ax=ax,
                         x=adjusted_index,
                         y=week_data[col],
                         label=label,
                         **kwargs)

    def plot_one_week(self, week, columns, **kwargs):
        """
        Plot line graphs for specified columns in a given week.
    
        Parameters:
            week (int): The week number to plot.
            columns (list or list of lists): A list of column names to be plotted. 
                If a list of lists is provided, each nested list will be plotted
                together on the same plot.
            **kwargs: Additional keyword arguments for the seaborn lineplot.
        """    
        if all(isinstance(col, list) for col in columns):
            subsets = columns
        else:
            subsets = [[col] for col in columns if col in self.data.columns]

        fig, axes = plt.subplots(len(subsets), 1,
                                 figsize=kwargs.get("figsize", (20, len(subsets) * 4)),
                                 squeeze=False)
        axes = axes.flatten()

        for i, subset in enumerate(subsets):
            for col in subset:
                self._plot_week_data(axes[i],
                                     week, col,
                                     label=col,
                                     **kwargs)
            axes[i].set_title(f"Week {week}, {' & '.join(subset)}")
            axes[i].set_xlabel("")
            axes[i].legend()

        plt.tight_layout()
        plt.show()

    def plot_many_weeks(self, weeks, columns, single_plot=True, **kwargs):
        """
        Plot line graphs for specified columns across multiple weeks.

        Parameters:
            weeks (list): A list of week numbers to plot.
            columns (list): A list of column names to be plotted.
            single_plot (bool): If True, all weeks are plotted on a single graph
                for each column.
            **kwargs: Additional keyword arguments for the seaborn lineplot.
        """
        if single_plot:
            fig, axes = plt.subplots(len(columns), 1,
                                     figsize=kwargs.get("figsize", (20, len(columns) * 4)),
                                     squeeze=False)
            axes = axes.flatten()

            for i, col in enumerate(columns):
                ax = axes[i]
                for week in weeks:
                    self._plot_week_data(axes[i],
                                         week, col,
                                         label=f"Week {week}",
                                         sharex_using_weekdays=True,
                                         **kwargs)
                ax.set_xlabel("")
                ax.set_title(col)
                ax.legend()
                ax.xaxis.set_major_locator(mdates.DayLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%a"))
        else:
            total_plots = len(weeks) * len(columns)
            fig, axes = plt.subplots(total_plots, 1,
                                     figsize=kwargs.get("figsize", (20, total_plots * 4)),
                                     squeeze=False)
            axes = axes.flatten()

            for i, (week, col) in enumerate(product(weeks, columns)):
                self._plot_week_data(axes[i],
                                     week,
                                     col,
                                     label=f"Week {week}",
                                     **kwargs)
                axes[i].set_xlabel("")
                axes[i].set_title(f"Week {week}, {col}")
                axes[i].legend()
        plt.tight_layout()
        plt.show()

    def plot_heatmap(self, weeks, col, **kwargs):
        """
        Generate heatmaps for a specified column across multiple weeks.

        Parameters:
            weeks (list): A list of week numbers for which heatmaps are to be generated.
            col (str): The column name for which the heatmap is to be generated.
            **kwargs: Additional keyword arguments for the heatmap.
        """
        for week in weeks:
            week_data = self._get_week_data(week, col)
            if week_data is not None:
                grouped_data = (week_data
                                .groupby(["day_of_week", "hour"])[col]
                                .sum()
                                .unstack())
                
            order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"]
            grouped_data = grouped_data.reindex(order)

            plt.figure(figsize=kwargs.get("figsize", (20, 2.5)))
            sns.heatmap(grouped_data,
                        yticklabels=([day[:3] for day in order]),
                        cbar_kws={"label": "[Wh]"},
                        cmap=kwargs.get("cmap", "Blues"),
                        **kwargs)
            plt.title(f"Week {week}, {col}")
            plt.yticks(fontsize=10)
            plt.xticks(fontsize=10)
            plt.ylabel("Day of week")
            plt.xlabel("Hour of day")
        plt.show()