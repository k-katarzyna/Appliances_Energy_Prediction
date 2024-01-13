import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.size"] = 14.0
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def histplots_grid(n_rows, n_cols, data, features = None):

    if features is None:
        features = [feature for feature in data.select_dtypes([int, float]).columns]
    
    width = n_cols * 3.2
    height = n_rows * 2.4
    
    fix, axes = plt.subplots(n_rows, n_cols, figsize = (width, height))
    
    if n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    for i, feature in enumerate(features):
        row, col = divmod(i, n_cols)
        
        if row >= n_rows:
            break
        
        ax = axes[row, col]
        sns.histplot(data[feature], ax = ax)
        ax.set_title(feature)
        ax.set_xlabel(None)
        ax.set_ylabel(None) 

    for j in range(i + 1, n_rows*n_cols):
        row, col = divmod(j, n_cols)
        axes[row, col].axis("off")

    plt.tight_layout()    
    plt.show()
    
    
def energy_consumption_all_time(appliances):

    daily_data = appliances.resample("D").mean()
    heatmap_data = daily_data.groupby([daily_data.index.day, daily_data.index.month]).mean().unstack(level=0)
    monthly_means = heatmap_data.mean(axis=1).to_frame("mean").astype(int)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 4), gridspec_kw = {"width_ratios": [30, 1]})

    vmin = heatmap_data.min().min()
    vmax = heatmap_data.max().max()
    months = ["Jan", "Feb", "Mar", "Apr", "May"]

    sns.heatmap(heatmap_data, cmap = "Blues", ax = ax1, vmin = vmin, vmax = vmax, cbar_kws = {"label": "[Wh]"})
    ax1.set_title("Daily Average Appliances Usage")
    ax1.set_xlabel("Day of Month")
    ax1.set_ylabel("Month")
    ax1.set_yticklabels(months, rotation = 0)
    ax1.set_xticks(np.arange(31) + .5)
    ax1.set_xticklabels(range(1, 32))

    sns.heatmap(monthly_means, cmap = "Blues", cbar = False, annot = True, fmt = "d", ax = ax2, vmin = vmin, vmax = vmax)
    ax2.set_title("Monthly Means")
    ax2.set_ylabel(None)
    ax2.set_yticklabels(months, rotation = 0)
    ax2.set_xticks([])

    plt.tight_layout()
    plt.show()
    
    
def energy_vs_lights_plot(appliances, lights):
    fig, ax1 = plt.subplots(1, 1, figsize=(20, 4))
    ax1.set_xlabel('Date')

    x_labels = appliances.index.strftime('%Y-%m-%d')
    x_labels = x_labels[::int(len(x_labels) / 6)]

    sns.lineplot(x=appliances.index, y=appliances, ax=ax1, color="midnightblue", linestyle="-", label="Appliances energy consumption")
    ax1.set_ylabel('Appliances [Wh]')
    ax1.set_title("General energy consumptions vs. lights energy consumption")
    ax1.set_xticks(x_labels)
    ax1.set_xticklabels(x_labels)

    ax2 = ax1.twinx()
    sns.lineplot(x=lights.index, y=lights, ax=ax2, color="skyblue", linestyle="-", label="Light energy consumption", alpha=0.8)
    ax2.set_ylabel('Lights [Wh]')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')
    plt.show()
    
    
def consumption_by_day_and_hour(data):

    grouped_data = data.groupby(["day_of_week", "hour"])["Appliances"].mean().unstack()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    grouped_data = grouped_data.reindex(order)

    plt.figure(figsize = (20, 5))
    sns.heatmap(grouped_data, cmap="Blues",  cbar_kws = {"label": "[Wh]"})

    plt.title("Average Appliances Usage by Day of Week and Hour")
    plt.ylabel("Day of Week")
    plt.yticks(ticks = np.arange(7) + .5, labels = [day[:3] for day in order])
    plt.xlabel("Hour of Day")

    plt.tight_layout()
    plt.show()