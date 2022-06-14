
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
import imageio

from datetime import timedelta
from pathlib import Path 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import keras

from utils import get_the_daily_data, create_x_y_datasets, \
            months_to_dict, predict_future, \
                plot_future, plot_future_evaluate, create_df_prediction
                    
from models import create_stacked_lstm

plot_folder = Path('/Users/gulcinvardar/Desktop/Data_Science_Bootcamp/stationary-sriracha-student-code/projects/week_final/plots/univariate/univariate_clean') 
csv_folder = Path('/Users/gulcinvardar/Desktop/Data_Science_Bootcamp/stationary-sriracha-student-code/projects/week_final/csvs/univariate') 
model_folder = Path('/Users/gulcinvardar/Desktop/Data_Science_Bootcamp/stationary-sriracha-student-code/projects/week_final/models') 
filepath_pred = Path('/Users/gulcinvardar/Desktop/Data_Science_Bootcamp/stationary-sriracha-student-code/projects/week_final/plots/univariate/univariate_preds')

features = ['temperature_air_mean_200']

start_date= "1981-01-01"
end_date= "2020-01-30"

df_daily = get_the_daily_data(start_date, end_date)

df_model = df_daily.iloc[:len(df_daily)-30]
df_holdout = df_daily.iloc[len(df_daily)-30:]

df_model_feature = df_model[features]

scaler = StandardScaler()
df_m = scaler.fit_transform(df_model_feature)


epochs = 150
batchsize = 1
neuron = 32
images_future = []
timesteps = [1, 3, 5, 7, 15, 30]


for timestep in timesteps:
    plotname = f'Timesteps: {timestep}'
    filename = f'S_univariate_forty_LSTM_timestep_{timestep}'
    X, y = create_x_y_datasets(df_m, timestep)
    X_train, X_test = train_test_split(X, test_size=0.05, shuffle=False)
    y_train, y_test = train_test_split(y, test_size=0.05, shuffle=False)
    model, history = create_stacked_lstm(X_train, y_train, epochs, batchsize, neuron)
    model.save(f'{model_folder}/{filename}_model.h5')
    plt.clf()
    plt.plot(pd.DataFrame(history.history))
    plt.savefig(f'{plot_folder}/{filename}_model_summary.jpg')
    prediction = predict_future(X_test, scaler, model)
    df_future = create_df_prediction(df_holdout, prediction)
    df_future.to_csv(f'{csv_folder}/{filename}_future.csv')
    plt.clf()
    plot_future(df_future, plotname)
    file = f'{plot_folder}/{filename}_futureplot.jpg'
    plt.savefig(file)
    images_future.append(imageio.imread(file))
imageio.mimsave(f'{plot_folder}/univariate_LSTM_future.gif', images_future, fps=2)


months_dictionary = months_to_dict()


for key, value in months_dictionary.items():
    plotname = value
    date_month = key
    start_date_test= "2020-01-30"
    end_date_test= f"2021-{date_month}-28"
    df_daily_test = get_the_daily_data(start_date_test, end_date_test)
    
    df_model_test = df_daily_test.iloc[:len(df_daily_test)-30]
    df_holdout_test = df_daily_test.iloc[len(df_daily_test)-30:]
    
    df_test_feature = df_model_test[[features]]
    df_test_feed = scaler.transform(df_test_feature)

    for timestep in timesteps:
        filename = f'S_univariate_forty_LSTM_timestep_{timestep}'
        X, y = create_x_y_datasets(df_test_feed, timestep)
        model = keras.models.load_model(f'{model_folder}/{filename}_model.h5')
        temp_prediction = predict_future(X, scaler, model)
        df_future = create_df_prediction(df_holdout_test, temp_prediction)
        plt.clf()
        plot_future_evaluate(df_future, timestep, plotname, filepath_pred)
