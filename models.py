import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import imageio
import calendar

from pathlib import Path 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, TimeDistributed
from keras.layers import LSTM, Conv1D, Flatten
from keras.optimizers import Adam

from wetterdienst import Settings
from wetterdienst.provider.dwd.observation import DwdObservationRequest, \
    DwdObservationDataset, DwdObservationPeriod, DwdObservationResolution



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

