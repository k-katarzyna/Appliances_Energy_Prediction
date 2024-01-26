## The goal

This project aims to develop a predictive model using multivariate time series analysis, focusing on household energy consumption. The model will utilize historical data on aggregated appliances' energy usage, temperature, humidity, and weather conditions to forecast 24-hour energy demand.

This approach differs from the original study [1], which explored various predictors for energy demand from which the dataset was sourced. The model could potentially contribute to the enhancement of Home Energy Management Systems (HEMS) by optimizing energy consumption, adapting to variable energy tariffs and integrating renewable energy sources like photovoltaics. Incorporation of residents' demand routines could make predictions more personalized. Such optimization is particularly relevant for smart homes, aiming to balance energy efficiency with user convenience, while also reducing their carbon footprint. [3]

The primary challenge lies in the limited dataset duration (4.5 months), which may affect the ability to predict seasonal variations in energy usage, thereby impacting the reliability and predictability of the model's forecasts.


## The dataset

The dataset, documenting appliance energy usage in a low-energy building, spans approximately 4.5 months. It records the energy consumption of appliances at 10-minute intervals. This dataset includes detailed information on the temperature and humidity levels in various rooms of the house, as well as outside, alongside data on lighting energy usage. Additionally, weather data from the nearest weather station has been integrated with the experimental data, synchronized by date and time. To evaluate regression models and pinpoint non-predictive attributes, the dataset also incorporates two random variables. [1, 2]


## References

1. Luis M. Candanedo, Veronique Feldheim, Dominique Deramaix, Data driven prediction models of energy use of appliances in a low-energy house, Energy and Buildings, Volume 140, 1 April 2017, Pages 81-97, ISSN 0378-7788. [Web Link](https://www.sciencedirect.com/science/article/abs/pii/S0378778816308970?via%3Dihub) (accessed 23.01.24)
2. https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction (accessed 23.01.24)
3. https://flixenergy.pl/blog-1388-inteligentne-systemy-zarzadzania-energia-w-naszym-domach (accessed 23.01.24)
4. https://en.wikipedia.org/wiki/Belgium#Geography (accessed 20.01.24)