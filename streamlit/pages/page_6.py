import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def page6():
    st.title("Facebook Prophet")
    st.subheader("An open source time-series analysis package")
    st.title("   ")

    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.subheader("Example Data Preparation")
        df = pd.read_csv('example_prophet_data.csv')
        st.write(df)
    
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")

    col1, col2, col3 = st.columns([1,6,1])
    with col1: 
        st.title("   ")
        st.title("   ")
        st.subheader('One year prediction based on 10 years data')
    with col2:
        st.image('pictures/yearly_prophet.jpg', 
            width=1000)
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")


    code = """from prophet import Prophet"""
    st.code(code, language='python')

    col1, col2, col3, col4, col5= st.columns([1,4, 1, 4, 1])

    with col2:

        code = """

model = Prophet()


model.fit(data)
future = model.make_future_dataframe(periods=360) 
prediction = model.predict(future)
model.plot(prediction)"""
        st.code(code, language='python')
       
    with col4:
        code = """
model = Prophet(growth='linear',
yearly_seasonality=False,
weekly_seasonality=False,
daily_seasonality=False,
holidays=None,
seasonality_mode='multiplicative',
seasonality_prior_scale=10,
holidays_prior_scale=0,
changepoint_prior_scale=0.01,
mcmc_samples=0
).add_seasonality(name='yearly',
period=365.25,
fourier_order=100,
prior_scale=10,
mode='multiplicative'
).add_seasonality(name='monthly', 
period=30.5, fourier_order=10)"""
        st.code(code, language='python')
    
    
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
    
    col1, col2, col3, col4, col5= st.columns([1, 2, 1, 2, 1])
    with col1:
        st.title("   ")
        st.title("   ")
        st.title("   ")
        st.title("   ")
        st.title("   ")
        st.subheader("Model Components")

    with col2:
        st.subheader("Simple Prophet")
        st.image('pictures/simple_prophet_components.jpg',
            width=500)
       
    with col4:
        st.subheader("Tweaked Prophet")
        st.image('pictures/complex_prophet_components.jpg',
            width=500)

    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
       
    col1, col2, col3, col4, col5= st.columns([1, 2, 1, 2, 1])
    with col1:
        st.title("   ")
        st.title("   ")
        st.subheader("Prediction of Unknown Future")

    with col2:
        st.subheader("Simple Prophet")
        st.image('pictures/predict_simple.jpg',
            width=500)
       
    with col4:
        st.subheader("Tweaked Prophet")
        st.image('pictures/predict_complex.jpg',
            width=500)

    'RMSE'