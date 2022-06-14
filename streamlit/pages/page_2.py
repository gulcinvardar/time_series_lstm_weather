import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px



def page2():

        st.title("Simple EDA")

        st.subheader("Get the data from Deutsche Wetter Dienst")
        st.subheader("from wetterdienst 0.37.0")

        code = """
        start_date = "1981-01-01"
end_date = "2020-01-01"  

result = DwdObservationRequest(
parameter=[DwdObservationDataset.CLIMATE_SUMMARY],
resolution=DwdObservationResolution.DAILY,
start_date=start_date,
end_date=end_date,
).filter_by_station_id(station_id=[433])
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

        st.subheader('Example daily dataset')
        df = pd.read_csv('example.csv')
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace= True)
        st.write(df)

        st.title("  ")
        st.title("  ")
        st.title("  ")
        st.title("  ")
        st.title("  ")
        st.title("  ")
        st.title("  ")
        st.title("  ")

        col1, col2 = st.columns([1, 1])
        with col1:
                st.subheader('Temperature (°C)')
                fig = px.scatter(df, x=df.index, y= 'temperature_air_mean_200')
                st.plotly_chart(fig)
        
        with col2:
                st.subheader('Humidity (%)')
                fig = px.scatter(df, x=df.index, y= 'humidity')
                st.plotly_chart(fig)

        st.title("  ")
        st.title("  ")
        st.title("  ")
        st.title("  ")
        st.title("  ")
        st.title("  ")
        st.title("  ")
        st.title("  ")
        
        col1, col2, col3, col4, col5  = st.columns([1, 1, 4, 1, 1])
        with col3:
                st.subheader('Correlation Heatmap')
                st.image('pictures/Heatmap_1.jpg',
                        width=640)
