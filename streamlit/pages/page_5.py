import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import base64


def page5():
    st.title("Multivariate LSTM with more features ")

    code = """
    start_date = "1981-01-01"
end_date = "2020-01-01"  
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
).filter_by_station_id(station_id=[433])"""

    st.code(code, language='python')

    code = """
    features = ['temperature_air_mean_200', 
                'month_sin', 'month_cos', 'day_sin', 'day_cos', 
                'temperature_dew_point_mean_200','humidity', 'humidity_absolute', 
                'pressure_vapor', 'wind_speed', 'sunshine_duration']
    """
    st.code(code, language='python')
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")

    st.image('pictures/paper_cover.png')

    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")


    col1, col2, col3, col4, col5= st.columns([1,2, 1, 2, 1])

    with col1:
        st.title("   ")
        st.title("   ")
        st.title("   ")
        st.title("   ")
        st.subheader("Prediction of Unknown Future")

    with col2:
        st.subheader("Multivariate LSTM")
        file_1 = open("pictures/multi_more_fourty.gif", "rb")
        contents_1 = file_1.read()
        data_url_1 = base64.b64encode(contents_1).decode("utf-8")
        file_1.close()
        st.header("   ")
        st.markdown(
                f'<img src="data:image/gif;base64,{data_url_1}" alt="testdata gif">',
                unsafe_allow_html=True,
                    )
       
    with col4:
        st.subheader("CNN-LSTM")
        file_2 = open("pictures/lstm_cnn.gif", "rb")
        contents_2 = file_2.read()
        data_url_2 = base64.b64encode(contents_2).decode("utf-8")
        file_2.close() 
        st.header("   ")
        st.markdown(
                f'<img src="data:image/gif;base64,{data_url_2}" alt="future gif">',
                unsafe_allow_html=True,
                )
        
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
    

    col1, col2, col3, col4, col5= st.columns([1,2, 1, 2, 1])
    with col1:
        st.title("   ")
        st.title("   ")
        st.title("   ")
        st.title("   ")
        st.subheader("RMSE")
    with col2:
        st.subheader("Multivariate LSTM")
        st.image('pictures/multi_more_fourty_rmse.jpg',
            width=500)
    with col4:
        st.subheader("CNN-LSTM")
        st.image('pictures/cnn_rmse.jpg',
            width=500)

    st.title("   ")
    st.title("   ")
    st.title("   ")
    st.title("   ")
