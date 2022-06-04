Time Series:

    - ARIMA
    -- This is usually with one variable
    -- excisionist variables
    -- So, you take only the Temperature



    - PROPHET





    - LSTM
    -- Hourly data might be better







General remarks:

    - To include the other features, use LSTM s
    - Start with lower number of input data
        -- when you increase the input your data must get more complex
        -- The evaulations might differ for short-term and long-term predictions
        -- Loss function MSE is good for evaluation because you can use the same unit
        -- if you do hourly data be careful with the evaluation
        -- gradient descent
        -- lattitude, longitude: for feature engineering 
            --- you can calculate the distance or the grid 
            --- pick one center for the enlem boylam
            --- angular windspeed???? direction?? As a good faeture engineering
            --- Search for feature engineering weather data


Combination of Sin and Cosine functions for cyclic features, instead of lag features 

linear regression history of 2 or 3 will be different



Try to add this to the script
# seasons = ['winter', 'spring', 'sommer', 'autumnn']
# seasons_start = {'winter':'2020-01-01', 'spring': '2020-04-01', 'sommer':'2020-07-01', 'autumnn':'2020-10-01'}
# seasons_end = {'winter':'2020-12-31', 'spring': '2021-03-31', 'sommer':'2021-06-31', 'autumnn':'2021-09-30'}
# for season in seasons:
#     start_date = seasons_start.get(season)
#     end_date = seasons_end.get(season)
#     df = get_the_daily_data(start_date, end_date)
#     scaler, scaled_input, df_input = scale_the_variable(df)