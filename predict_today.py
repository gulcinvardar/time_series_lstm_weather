
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from pathlib import Path 


from sklearn.preprocessing import StandardScaler


import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.optimizers import Adam


from wetterdienst import Settings
from wetterdienst.provider.dwd.observation import DwdObservationRequest, \
    DwdObservationDataset, DwdObservationPeriod, DwdObservationResolution


def convert_to_celsius(df):
    """
    Convert the unit of temperature 
    from Kelvin to Celsius for all the columns.
    """
    col_names = df.filter(regex='temperature').columns
    df[col_names] = df[col_names].sub(273.15)

    return df


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


def insert_end(Xin,new_input, timestep):
    """Insert the prediction into the input data."""
    for i in range(timestep-1):
        Xin[:,i,:] = Xin[:,i+1,:]
    Xin[:,timestep-1,:] = new_input
    
    return Xin

    
def get_time_cosine(df):
    df['month'] = df.index.month
    df['day'] = df.index.day_of_year
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_sin'] = np.sin(2 * np.pi * df['day']/365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/365.25)

    return df

def predict_future(X, scaler, model, timestep):
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

model_folder = Path('/Users/gulcinvardar/Desktop/Data_Science_Bootcamp/stationary-sriracha-student-code/projects/week_final/models') 
start_date= "2001-01-01"
end_date= "2020-01-01"
df = get_the_daily_data(start_date, end_date)

df_model = df.iloc[:len(df)-30]
df_holdout = df.iloc[len(df)-30:]

df_model = get_time_cosine(df_model)

features = ['temperature_air_mean_200', 'month_sin', 'month_cos', 'day_sin', 'day_cos']
df_model_feature = df_model[features]


scaler = StandardScaler()
df_m = scaler.fit_transform(df_model_feature)


start_date_test= "2022-03-30"
end_date_test= f"2022-06-07"
df_daily_test = get_the_daily_data(start_date_test, end_date_test)
df_model_test = get_time_cosine(df_daily_test)
df_test_feature = df_model_test[features]
df_test_feed = scaler.transform(df_test_feature)
X, y = create_x_y_datasets(df_test_feed, 5)
model = keras.models.load_model(f'{model_folder}/S_multi_years_twenty_time_LSTM_timestep_5_model.h5')
temp_prediction = predict_future(X, scaler, model, 5)

prediction_df = pd.DataFrame({
    'date':pd.date_range(start=end_date_test, periods=31)
})[1:]
prediction_df['prediction'] = temp_prediction
prediction_df.to_csv('final_prediction_lstm.csv')
