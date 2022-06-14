import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import calendar 


import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, TimeDistributed
from keras.layers import LSTM, Conv1D, Flatten
from keras.optimizers import Adam

from wetterdienst import Settings
from wetterdienst.provider.dwd.observation import DwdObservationRequest, \
        DwdObservationDataset, DwdObservationResolution


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

def get_the_hourly_data(start_date, end_date):
    """
    Get the hourly data from DWD and do initial cleaning.
    """
    Settings.tidy = True
    Settings.humanize = True
    request = DwdObservationRequest(
        parameter=[
            DwdObservationDataset.TEMPERATURE_AIR,
            DwdObservationDataset.TEMPERATURE_SOIL, 
            DwdObservationDataset.DEW_POINT, 
            DwdObservationDataset.VISIBILITY, 
            DwdObservationDataset.SUN,
            DwdObservationDataset.PRESSURE,
            DwdObservationDataset.WIND,
            DwdObservationDataset.MOISTURE,
            DwdObservationDataset.PRECIPITATION],
        resolution=DwdObservationResolution.HOURLY,
        start_date=start_date,
        end_date=end_date,
        ).filter_by_station_id(station_id=[433])
    df = request.values.all().df

    return df

def convert_to_celsius(df):
    """
    Convert the unit of temperature 
    from Kelvin to Celsius for all the columns.
    """
    col_names = df.filter(regex='temperature').columns
    df[col_names] = df[col_names].sub(273.15)

    return df

def get_time_cosine(df):
    """Get the sine and cosine of the month and the day"""
    df['month'] = df.index.month
    df['day'] = df.index.day_of_year
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_sin'] = np.sin(2 * np.pi * df['day']/365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/365.25)

    return df

def clean_and_resample(df):
    """
    Convert all the temperature info from Kelvin to Celsius.
    """
    df_pivot = df.pivot_table(values='value', index='date', columns='parameter')
    df_pivot.index =  pd.to_datetime(df_pivot.index, format="%Y-%m-%d %H:%M:%S")
    df_resample = df_pivot.resample('D').max()
    df_interp = df_resample.interpolate()
    df_clean = convert_to_celsius(df_interp)

    return df_clean


def create_x_y_datasets(df, timestep):
    """
    Create input and output data based on timestep.
    Split the data as train and test.
    """
    X = []
    y = []

    for i in range(len(df) - (timestep)):
        X.append(df[i:i+timestep])
        y.append(df[i+timestep])

    X=np.array(X)
    y=np.array(y)


    return X, y


def earlystop():
    """Stop the learning based on val_loss."""
    early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    min_delta=0.0005, 
    patience=10, 
    verbose=1, 
    )
    
    return early_stop


def create_stacked_lstm(X, y, epochs, batchsize, neuron):
    """Build and fit the model with multiple layer."""
    K.clear_session()
    model = Sequential()
    model.add(LSTM(neuron*4, input_shape= (X.shape[1], X_train.shape[-1]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(neuron*2, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(neuron, return_sequences=False))
    model.add(Dense(y.shape[-1], activation='linear'))
    opt = keras.optimizers.Adam(learning_rate=0.001) # play around with this
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    earlystopping = earlystop()
    history = model.fit(X, y, batch_size=batchsize, epochs=epochs, validation_split=0.2, callbacks = [earlystopping], verbose=2)

    return model, history

def create_cnn_lstm(X, y, epochs, batchsize, timestep):
    """Build and fit the model with CNN and double layer LSTM."""
    K.clear_session()
    model = Sequential()
    model.add(TimeDistributed(Conv1D(8, kernel_size=(3), strides=(1), padding='same', activation="relu"), input_shape=(None, timestep, 11)))
    model.add(TimeDistributed(Conv1D(8, kernel_size=(2), strides=(1), padding='same', activation="relu")))
    model.add(TimeDistributed(Conv1D(16, kernel_size=(2), strides=(1), padding="same", activation="relu")))
    model.add(TimeDistributed(Conv1D(16, kernel_size=(3), strides=(1), padding="same", activation="relu")))
    model.add(TimeDistributed(Conv1D(32, kernel_size=(3), strides=(1), padding="same", activation="relu")))
    model.add(TimeDistributed(Conv1D(64, kernel_size=(2), strides=(1), padding="same", activation="relu")))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(24, return_sequences=True))
    model.add(LSTM(2))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='linear'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='linear'))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[-1], activation='linear'))
    opt = keras.optimizers.Adam(learning_rate=0.001) 
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    earlystopping = earlystop()
    history = model.fit(X, y, batch_size=batchsize, epochs=epochs, validation_split=0.2, callbacks = [earlystopping], verbose=2)

    return model, history


def insert_end(Xin,new_input, timestep):
    """Insert the prediction into the input data."""
    for i in range(timestep-1):
        Xin[:,i,:] = Xin[:,i+1,:]
    Xin[:,timestep-1,:] = new_input
    
    return Xin

def insert_end_cnn(Xin,new_input, timestep):
    """
    Insert the prediction into the input data.
    Shape is different than LSTM prediction.
    """
    for i in range(timestep-1):
        Xin[:,:,i,:] = Xin[:,:,i+1,:]
    Xin[:,:,timestep-1,:] = new_input
    
    return Xin


def months_to_dict():
    """
    Create a dictionary for numerical and alphabetical months.
    """
    months = list(calendar.month_name)[1:]
    month_number = ["%.2d" % i for i in range(13)][1:]
    months_dictionary = dict(zip(month_number, months))

    return months_dictionary

def create_df_prediction(df, prediction):
    """Get the prediction into a dataframe."""
    df_future = df[['temperature_air_mean_200']]
    df_future['prediction'] = prediction

    return df_future

def predict_future(X, scaler, model, timestep):
    """Predict future one by one."""
    X_in = X[-1:,:,:]
    prediction_test = np.empty((0,y.shape[-1]), float)
    for day in range(30):
        X_out = model.predict(X_in, batch_size=1)
        prediction_test = np.append(prediction_test, X_out, axis=0) 
        X_in = insert_end(X_in, X_out[0], timestep)
    prediction_rescale = scaler.inverse_transform(prediction_test)
    temp_prediction = []
    for temp in range(len(prediction_rescale)):
        temp_prediction.append(prediction_rescale[temp, 0])

    return temp_prediction

def predict_future_cnn(X, scaler, model, timestep):
    """
    Predict future one by one for CNN-LSTM.
    Shape is different than the predict future method.
    """
    X_in = X[-1:,:,:,:]
    prediction_test = np.empty((0,y.shape[-1]), float)
    for day in range(30):
        X_out = model.predict(X_in, batch_size=1)
        prediction_test = np.append(prediction_test, X_out, axis=0) 
        X_in = insert_end_cnn(X_in, X_out[0], timestep)
    prediction_rescale = scaler.inverse_transform(prediction_test)
    temp_prediction = []
    for temp in range(len(prediction_rescale)):
        temp_prediction.append(prediction_rescale[temp, 0])

    return temp_prediction

def plot_future(df, plotname):
    """Plot the future"""
    plt.plot(df['prediction'])
    plt.plot(df['temperature_air_mean_200'])
    plt.xticks(rotation=45)
    plt.xlabel('Future Date')
    plt.ylabel('Temperature (C)')
    plt.legend(('Prediction','Actual'))
    plt.title(plotname, fontsize=18)

def plot_future_evaluate(df, timestep, plotname, filepath):
    """Plot and save the future."""
    plt.plot(df['prediction'])
    plt.plot(df['temperature_air_mean_200'])
    plt.xticks(rotation=45)
    plt.xlabel('Future Date')
    plt.ylabel('Temperature (C)')
    plt.ylim(-15, 30)
    plt.legend(('Prediction','Actual'))
    plt.title(f'{plotname}: Timestep {timestep}', fontsize=18)
    plt.savefig(f'{filepath}/{plotname}_Timestep_{timestep}.jpg')

def evaluate_prophet(df, start_date_test, prediction):
    """Evaluate prophet's prediction."""
    time = pd.DataFrame({
    'date':pd.date_range(start=start_date_test, periods=30)
    })
    ytrue = df['temperature_air_mean_200']
    time['ytrue'] = ytrue.tolist()
    yhat = prediction['yhat'].iloc[len(prediction) - 30 : ]
    time['yhat'] = yhat.tolist()
    time.set_index('date', inplace = True)
    plt.plot(time.index, time['yhat'])
    plt.plot(time.index, time['ytrue'])
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Temperature (C)')
    plt.legend(('Prediction', 'Actual'))

    return time

    