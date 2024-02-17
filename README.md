## This project is in progress...
This project aims to develop a predictive model using multivariate time series analysis.\
Done so far:
1. Exploratory data analysis (main insights presented in the *0_data_exploration.ipynb* notebook)
2. Determination of reference points (see *1_experiments.ipynb* notebook) and initial tests(not presented)
3. Experiments (see *1_experiments.ipynb* notebook)
   * Feature engineering
     - lagged and window features for target variable and others
     - interaction features for weather, house microclimate and datetime attributes
     - recursive feature elimination
   * Sample weighting

## The dataset
The dataset documents appliance energy usage in a low-energy building in Belgium. The house, with a floor space of 280 m2, is equipped with a heat pump for water heating, a fireplace heating system and a heat recovery ventilation system. It is inhabited by four occupants (two adults and two teenagers), with one adult working from home. The dataset includes information on the temperature and relative humidity levels in various rooms of the house, as well as outside, along with data on lighting energy usage. The data at the house was collected at 10-minute intervals, using M-BUS energy counters for appliances and lights data, and ZigBee sensor network for microclimate information. Additionally, weather data from the nearest weather station was integrated into this experimental dataset, synchronized by date and time. Since the data from the station were available hourly, linear interpolation was applied to fill in the gaps, maintaining the 10-minute interval consistency. To evaluate regression models and identify non-predictive attributes, the dataset also includes two random variables [1, 2].

## The goal
The objective is a preliminary exploratory investigation into whether efficient forecasting of 24-hour energy demand is feasible with data of this nature, experimenting with different feature engineering and model types. The project is driven by a personal interest in working with and learning about time series data.\
This approach differs from the original study from which the dataset was obtained [1], where the impact of various factors on energy demand was explored, but not using time series forecasting methodology, rather just for regression analysis. Based on datetime information, however, additional features were created to capture dependencies on the routine of the household members: day of the week, the number of minutes since midnight and the status of the week (whether it is a weekend).

## Conceptual background
This type of model could potentially enhance Home Energy Management Systems (HEMS) for smart homes. The evolution of HEMS represents now not only systems operating according to predefined settings, there are also self-learning systems powered by AI, capable of adjusting to the routine of users, providing even greater convenience, automation, but also savings and a reduction in the carbon footprint. They aid in optimizing energy consumption, adapting to variable energy tariffs or integrating renewable energy sources like photovoltaics [3].\
Although one can read about optimization by such systems mainly for things like air and water temperature regulation or lighting, or whether they are used in managing energy in buildings in distributed renewable energy systems and not in a single home, if the energy consumption of devices is also linked with various external factors [1] and the routine of the inhabitants, such a prediction seems interesting to test. Offering personalized insights, based not only on external factors and previous settings but also on the known routines of users, possibly this prediction could make energy management more automated if needed or just give better recommendation for using very energy-intensive devices. The proposed timeframe provides a practical opportunity to consider dynamically changing factors (such as weather conditions), allowing users to react or plan eventual actions if necessary.

**The primary challenge** is the limited dataset duration (4.5 months), which may affect the ability to predict seasonal variations in energy usage, thereby affecting the model's forecast reliability and predictability.\
**Another challenge** is that household behaviors cannot be entirely predicted and depend on many factors, not only those included in the data. Even included factors, like weather, cannot be entirely forecasted.\
Such data would need to be regularly updated and the model should adapt to new information. But the model should be primarily accurate, with operational speed being a secondary concern, as well as its complexity. However, it shouldn't be too complex so that end-users can understand the basis of its predictions, which may be essential in adhering to recommendations. Assuming avoiding large errors that would translate into higher costs associated, e.g., either with storing energy or with unnecessary use of energy from the grid, the RMSE metric will be used as the main, so that large errors will have greater weight in evaluating the model, but to still have a more tangible idea of the size of the error than with MSE.

## References
1. Luis M. Candanedo, Veronique Feldheim, Dominique Deramaix, Data driven prediction models of energy use of appliances in a low-energy house, Energy and Buildings, Volume 140, 1 April 2017, Pages 81-97, ISSN 0378-7788. [Web Link](https://www.sciencedirect.com/science/article/abs/pii/S0378778816308970?via%3Dihub) (accessed 23.01.24)
2. https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction (accessed 23.01.24)
3. https://flixenergy.pl/blog-1388-inteligentne-systemy-zarzadzania-energia-w-naszym-domach (accessed 23.01.24)