# Comparison of different LSTM models and Prophet for Time-Series Weather Forecast

This project was developed as the final project for Spiced Academy Data Science Bootcamp. 


The traditional approach for time-series analysis employes numerical analysis such as ARIMA. 
The main aim of this project is to create an LSTM time-series model most suitable for weather forecast. 
Throughout the whole project, the variable for the prediction (output) is the mean air temperature. 
The analysis is done for Berlin Tempelhof and the data is extracted from Deutsche Wetter Dienst. 
For how to use the Wetterdienst Package, you can have a look at the files wetter_dienst_station.ipynb.  
However for a detailed documentation, please refer to  [earthobservations / wetterdienst](https://github.com/earthobservations/wetterdienst). 

## Main approach 
In an LSTM time-series model, the selection of the time window to include in the model training is one of the most important factors.
Therefore, the efficacy of different time-windows in model training are compared for:

1. Univariate LSTM that includes only the past air temperature.
2. Multivariate LSTM that includes the month and the day of the year as features. 
Because of their cyclic nature, the time related features were included both as sine and cosine functions.
3. Multivariate LSTM that includes additional features such as pressure vapor, humidity, sunshine duration.
4. Multivariate CNN-LSTM that includes the same as in (3)
5. Univariate Prophet.

For all the LSTM models, the testing of the time-window is automatized in a for loop. 
All the same models were also tested with the models being trained with 10 years, 20 years, or 40 years daily data.

## Inadequacy of the prediction of the validation data for model evaluation

In most of the LSTM models found online for Time-Series, the prediction of the validation data is used as a measure of model accuracy. 
However, this approach remains short on visualzing how well the model perform in a multi-step prediction of the unknown future.
For that reason, not the validation data but the holdout data taken as unknown future was used for model evaluation in this project. 
The unknown future was predicted for 30 days. 

### Prediction of the validation data
![prediction-of-validation-data](https://github.com/gulcinvardar/time_series_lstm_weather/blob/main/streamlit/pages/pictures/S_LSTM_timestep_testdata.gif)

### Prediction of the validation data
![prediction-of-future-data](https://github.com/gulcinvardar/time_series_lstm_weather/blob/main/streamlit/pages/pictures/univariate_S_LSTM_uni_future.gif)


## Results
Ultimately, the LSTM model that includes the time as features and that is trained with 20 years of daily data yielded the lowest RMSE.
The best time-window to be used for Time-Series Analysis is 5.

### Prediction of the validation data
![prediction-of-future-data](https://github.com/gulcinvardar/time_series_lstm_weather/blob/main/streamlit/pages/pictures/S_multi_years_twenty_time_LSTM_future.gif)

### Comparison of the different models trained with 5-days time window
![compare-the-models](https://github.com/gulcinvardar/time_series_lstm_weather/blob/main/streamlit/pages/pictures/timestep_5_compare.jpg)

## The contents
- a wetterdienst file to find the Berlin stations among German stations.
- a utils file with helper functions to clean the data, perform feature engineering, and to plot and evaluate the results 
- a models file with LSTM and CNN-LSTM models
- a very simple EDA of weather data for Berlin Tempelhof. Please note, some features are available only for hourly data. 
Those features were then resampled as the max Temperature with a daily frequency.
- four different LSTM files to test different time-windows for each model. 
The files include big chuncks of the same code that will be cleaned up soon. 
- a Prophet file to play around with Prophet parameters. To use Prophet, please follow [Prophet-instructions](https://facebook.github.io/prophet/).
- a separate file to calculate the RMSE using CSV files saved during model training.
- a file for the prediction of the real future days. 

The same LSTM files were used to train the models with 10, 20, or 40 years of daily data. 
For each training, the start date for the request of historical weather information was changed accordingly.

## Outlook
- Include more features such as other weather stations, altitude, longitude etc.
- Try logaritmic scaling of the features instead of Standard Scaling.
- Train the models with hourly data instead of daily data. Predict hourly and then resample for daily temperature prediction. 
This might not work for the prediction of several days in the future as that means more hourly data to predict.
- Include satellite images??


### License

(c) 2022 Gülçin Vardar

Distributed under the conditions of the MIT License. See LICENSE for details.