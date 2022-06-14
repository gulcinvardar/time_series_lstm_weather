import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import base64


def get_time_cosine(df):
    df['month'] = df.index.month
    df['day'] = df.index.day_of_year
    df['year'] = df.index.year
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_sin'] = np.sin(2 * np.pi * df['day']/365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/365.25)

    return df

def get_example_df():
    df = pd.read_csv('all_data_resampled.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = get_time_cosine(df)

    return df

def page4():
    st.title("Multivariate LSTM with time info")


    df = get_example_df()

    st.title("  ")
    st.title("  ")
    
    col1, col2, col3, col4, col5= st.columns([1,2, 1, 2, 1])

    with col2:
        df = pd.read_csv('all_data_resampled.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = get_time_cosine(df)
        st.subheader('Day of the year')
        fig, ax = plt.subplots(figsize=(5,4))
        ax = sns.lineplot(x = df.index[300:666], y=df['day_sin'].iloc[300:666])
        plt.xticks(rotation = 45)
        st.pyplot(fig)

    with col4:
        df = pd.read_csv('all_data_resampled.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = get_time_cosine(df)
        st.subheader('Month of the year')
        fig, ax = plt.subplots(figsize=(5,4))
        ax = sns.lineplot(x = df.index[300:666], y=df['month_sin'].iloc[300:666])
        plt.xlabel('Date')
        plt.xticks(rotation = 45)
        st.pyplot(fig)

    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    

    code = '''def create_stacked_lstm(X, y, epochs, batchsize, neuron):
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
        history = model.fit(X, y, batch_size=batchsize, epochs=epochs, validation_split=0.2, callbacks = [earlystopping], verbose=2)'''

    st.code(code, language='python')

    code = """
    features = ['temperature_air_mean_200', 'month_sin', 'month_cos', 'day_sin', 'day_cos']
    """
    st.code(code, language='python')

    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    
    col1, col2, col3, col4, col5= st.columns([1,2, 1, 2, 1])
    with col1:
        st.title("  ")
        st.title("  ")
        st.title("  ")
        st.title("  ")
        st.title("  ")
        st.subheader("Compare Loss and Validation Loss")
    with col2:
        st.header("20 Years of data")
        st.image('pictures/twenty_5_model_summary.jpg',
            width=500)
    with col4:
        st.header("40 Years of data")
        st.image('pictures/fourty_5_model_summary.jpg',
            width=500)


    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")


    col1, col2, col3, col4, col5= st.columns([1,2, 1, 2, 1])

    with col1:
        st.title('   ')
        st.title('   ')
        st.subheader("Compare unknown future")
    with col2:
        st.header("20 Years of data")
        file_1 = open("pictures/multi_twenty.gif", "rb")
        contents_1 = file_1.read()
        data_url_1 = base64.b64encode(contents_1).decode("utf-8")
        file_1.close()
        st.markdown(
                f'<img src="data:image/gif;base64,{data_url_1}" alt="testdata gif">',
                unsafe_allow_html=True,
                    )
       
    with col4:
        st.header("40 Years of data")
        file_2 = open("pictures/S_multi_years_forty_time_LSTM_future.gif", "rb")
        contents_2 = file_2.read()
        data_url_2 = base64.b64encode(contents_2).decode("utf-8")
        file_2.close() 
        st.markdown(
                f'<img src="data:image/gif;base64,{data_url_2}" alt="future gif">',
                unsafe_allow_html=True,
                )
        
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")

    col1, col2, col3, col4, col5= st.columns([1,2, 1, 2, 1])
    with col1: 
        st.title('   ')
        st.title('   ')
        st.subheader("Compare RMSE")
    with col2:
        st.header("20 Years of data")
        st.image('pictures/multi_twenty_rmse.jpg',
            width=500)
    with col4:
        st.header("40 Years of data")
        st.image('pictures/times_true_fourty.jpg',
            width=500)

