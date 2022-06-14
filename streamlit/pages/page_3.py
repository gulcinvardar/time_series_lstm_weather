#import std libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import base64

def page3():
    
    col1, col2, col3 = st.columns([1,4,1])
    with col2:
        st.title(" Univariate LSTM ")

    col1, col2, col3 = st.columns([1,4,1])
    with col1:
        st.subheader('Single Timestep, One Year of data')
    with col2:
        st.image('pictures/single_layer_lstm.png', width=640)
    with col3:
        st.write('blue: actual')
        st.write('orange: prediction of the train')
        st.write('green: prediction of the validation')

    col1, col2, col3, col4, col5 = st.columns([1,1,2,1,1])
    with col3:
        st.subheader('Do you also see something wrong here?')


    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")

    code = """def create_x_y_datasets(df, timestep):
    X = []
    y = []

    for i in range(len(df) - (timestep)):
        X.append(df[i:i+timestep])
        y.append(df[i+timestep])

    X=np.array(X)
    y=np.array(y)

    return X, y"""
    
    st.code(code, language='python')

    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")


    file_1 = open("pictures/S_LSTM_timestep_testdata.gif", "rb")
    contents_1 = file_1.read()
    data_url_1 = base64.b64encode(contents_1).decode("utf-8")
    file_1.close()

    file_2 = open("pictures/univariate_S_LSTM_uni_future.gif", "rb")
    contents_2 = file_2.read()
    data_url_2 = base64.b64encode(contents_2).decode("utf-8")
    file_2.close()
    
    col1, col2, col3  = st.columns([1, 7, 1])
    with col2:
        st.header("Prediction of the test data as test data")
        st.markdown(
                f'<img src="data:image/gif;base64,{data_url_1}" alt="testdata gif">',
                unsafe_allow_html=True,
                    )

    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")
    st.title("  ")

    
    col1, col2, col3  = st.columns([1, 7, 1])
    with col2:
        st.header("Prediction of the test data as unknown future")
        st.markdown(
                f'<img src="data:image/gif;base64,{data_url_2}" alt="future gif">',
                unsafe_allow_html=True,
                )

