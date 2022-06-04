
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math
import imageio

from utils import get_the_hourly_data, get_the_daily_data

from datetime import timedelta
from pathlib import Path 

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM

from wetterdienst import Wetterdienst
from wetterdienst import Resolution, Period
from wetterdienst import Settings
from wetterdienst.provider.dwd.observation import DwdObservationRequest, DwdObservationDataset, DwdObservationPeriod, DwdObservationResolution



def convert_to_celsius(df):
    """
    Convert the unit of temperature 
    from Kelvin to Celsius for all the columns.
    """
    col_names = df.filter(regex='temperature').columns
    df[col_names] = df[col_names].sub(273.15)

    return df

    # bu aslinda utils'te, simdilik burda dursun


def get_the_daily_data(start_date, end_date):
    """
    Get the daily data from DWD and do initial cleaning.
    """
    Settings.tidy = False
    Settings.default()
    request = DwdObservationRequest(
        parameter=[DwdObservationDataset.CLIMATE_SUMMARY],
        resolution=DwdObservationResolution.DAILY,
        start_date=start_date,
        end_date=end_date,
        ).filter_by_station_id(station_id=[433])
    df = request.values.all().df
    df = convert_to_celsius(df)
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    df.set_index('date', inplace= True)

    return df

     # bu aslinda utils'te, simdilik burda dursun


def scale_the_variable(df):
    """Select the variable."""
    df_input = df['temperature_air_mean_200']
    scaler= MinMaxScaler(feature_range=(-1,1))
    scaled_input = scaler.fit_transform(pd.DataFrame(df_input))

    return scaler, scaled_input, df_input


def create_x_y_datasets(scaled_input, timestep, test_day_number):
    """
    Create input and output data based on timestep.
    Split the data as train and test.
    """
    X = []
    y = []

    for i in range(len(scaled_input) - (timestep)):
        X.append(scaled_input[i:i+timestep])
        y.append(scaled_input[i+timestep])

    X=np.array(X)
    y=np.array(y)

    X_train = X[:len(scaled_input) - test_day_number,:,:]
    X_test = X[len(scaled_input) - test_day_number:,:,:]    
    y_train = y[:len(scaled_input) - test_day_number]    
    y_test= y[len(scaled_input) - test_day_number:]

    return X_train, X_test, y_train, y_test


def create_vanilla_lstm(X, y, timestep, neuron, epochs):
    """Create the LSTM model with one layer."""
    K.clear_session()
    model = Sequential()
    model.add(LSTM(neuron, input_shape=(timestep, 1), use_bias=False))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    earlystopping = earlystop()
    history = model.fit(X, y, batch_size=1, epochs=epochs, validation_split=0.2, callbacks = [earlystopping], verbose=2)

    return model, history


def earlystop():
    """Stop the learning based on val_loss."""
    early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    min_delta=0.0005, 
    patience=10, 
    verbose=1, 
    )
    
    return early_stop


def create_stacked_lstm(X, y, timestep, epochs, batchsize, neuron=32):
    """Build and fit the model."""
    K.clear_session()
    model = Sequential()
    model.add(LSTM(neuron*4, input_shape= (timestep,1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(neuron*2, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(neuron, return_sequences=False))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    earlystopping = earlystop()
    history = model.fit(X, y, batch_size=batchsize, epochs=epochs, validation_split=0.2, callbacks = [earlystopping], verbose=2)

    return model, history


def predict_and_rescale(scaler, model, X, batch_size):
    """Predict the target values and rescale."""
    predict = model.predict(X, batch_size = batch_size)
    pred_rescale = scaler.inverse_transform(predict)
    y_pred = np.hstack(pred_rescale).tolist()[1:]

    return y_pred


def insert_end(Xin,new_input):
    """Insert the prediction into the input data."""
    for i in range(timestep-1):
        Xin[:,i,:] = Xin[:,i+1,:]
    Xin[:,timestep-1,:] = new_input
    
    return Xin


def predict_future(scaler, model, future, X_test, df_input):
    """
    Predict the future days.
    Create a dataframe with the prediction and the timestamp of the future.
    """
    forecast = []
    time=[]
    Xin = X_test[-1:,:,:]
    for i in range(future):
        out = model.predict(Xin, batch_size=1)    
        forecast.append(out[0,0]) 
        Xin = insert_end(Xin,out[0,0]) 
        time.append(pd.to_datetime(df_input.index[len(df_input)-test_day_number-1])+timedelta(days=i+1))
    
    forecast_array = np.array(forecast).reshape(-1,1) 
    future_pred = scaler.inverse_transform(forecast_array)
    future_list = np.hstack(future_pred).tolist()

    df_future = pd.DataFrame({'date': time, 'forecast': future_list}, columns=['date','forecast'])
    df_future.set_index('date', inplace=True)
    return df_future


def plot_and_rmse_future(filename, filepath, plotname, images, df_input, df_future):
    """
    Plot the future predictions vs real data.
    Calculate RMSE:
    """
    plt.figure(figsize=(16,8))
    plt.title(plotname, fontsize=18)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Temperature (C)', fontsize=18)
    plt.ylim(-10, 15)
    plt.plot(df_input[332:], linewidth=2)
    plt.plot(df_future,"r--",linewidth=2)
    plt.legend(('Actual','Predicted'), loc = "upper left")
    file = f'{filepath}/{filename}_future.jpg'
    plt.savefig(file)
    images.append(imageio.imread(file))

    future_score = mean_squared_error(df_input[len(df_input)-test_day_number:], df_future['forecast'])

    return future_score


def plot_and_rmse_test(filename, filepath, plotname, images, y_test_true, ytest_pred):
    """
    Plot the prediction of the test data vs real data.
    Calculate RMSE:
    """
    plt.figure(figsize=(20,9))
    plt.xlabel('number of days', fontsize=18)
    plt.ylabel('Temperature (C)', fontsize=18)
    plt.ylim(-10, 15)
    plt.xlim(-1, 33)
    plt.plot(y_test_true , 'blue', linewidth=5)
    plt.plot(ytest_pred,'r' , linewidth=4)
    plt.legend(('Test','Predicted'))
    plt.title(plotname, fontsize=18)
    file = f'{filepath}/{filename}_testdata.jpg'
    plt.savefig(file)
    images.append(imageio.imread(file))

    test_score = mean_squared_error(y_test_true[:-1], ytest_pred)

    return test_score


def plot_model(filename, filepath, plotname, history):
    """Plot the loss values of the model."""
    ax = pd.DataFrame(history.history).plot()
    ax.set_title(plotname)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    plt.savefig(f'{filepath}/{filename}_loss.jpg')


def evaluate_model(images_test, images_future):
    plot_model(filename, filepath, plotname, history)
    ytest_pred = predict_and_rescale(scaler, model, X_test, batch_size)
    y_test_true_rescale = scaler.inverse_transform(y_test)
    y_test_true = np.hstack(y_test_true_rescale).tolist()
    test_score = plot_and_rmse_test(filename, filepath, plotname, images_test, y_test_true, ytest_pred)
    df_future = predict_future(scaler, model, test_day_number, X_train, df_input)
    future_score = plot_and_rmse_future(filename, filepath, plotname, images_future, df_input, df_future)

    return test_score, future_score
   

start_date= "2021-01-01"
end_date= "2021-12-31"
df = get_the_daily_data(start_date, end_date)

scaler, scaled_input, df_input = scale_the_variable(df)

filepath = Path('/Users/gulcinvardar/Desktop/Data_Science_Bootcamp/stationary-sriracha-student-code/projects/week_final/plots/univariate') 
test_day_number = 33
neuron = 32
batch_size = 1
epochs = 100
times = []
test_rmse=[]
future_rmse = []
images_future = []
images_test = []

for i in range(1, 16):
    timestep = i
    times.append(i)
    filename = f'S_LSTM_timestep_{i}'
    plotname = f'Timesteps: {i}'
    X_train, X_test, y_train, y_test = create_x_y_datasets(scaled_input, timestep, test_day_number)
    model, history = create_stacked_lstm(X_train, y_train, timestep, epochs, batch_size, neuron)
    test_score, future_score = evaluate_model(images_test, images_future)
    test_rmse.append(test_score)
    future_rmse.append(future_score)
imageio.mimsave(f'{filepath}_S_LSTM_uni_future.gif', images_future, fps=2)
imageio.mimsave(f'{filepath}_S_LSTM_uni_test.gif', images_test, fps=2)

plt.clf()
plt.plot(times, test_rmse)
plt.plot(times, future_rmse)
plt.xlabel('timesteps')
plt.ylabel('RMSE')
plt.savefig(f'{filepath}/RMSE_stacked.jpg')