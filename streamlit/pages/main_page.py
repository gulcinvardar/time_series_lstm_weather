

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import base64


from page_2 import page2
from page_3 import page3
from page_4 import page4
from page_5 import page5
from page_6 import page6
from page_7 import page7



st.set_page_config(layout="wide")

def main_page():
    
    col1, col2, col3, col4 = st.columns([1, 1, 7, 1])
    with col3:
        st.title('Time-Series Weather Forecast')
        st.title('LSTM models')
        st.image('pictures/wedding_weather_bootcamp.jpg', 
                caption= 'An AI created image for rainy wedding',
                width=500)
        st.subheader("Is it going to rain at my wedding? 🎈")


    
page_names_to_funcs = {
    "Main Page": main_page,
    "Simple EDA": page2,
    "Univariate LSTM": page3,
    "Multivariate LSTM (Time)": page4,
    "More Features & CNN-LSTM": page5,
    "Prophet": page6,
    "Prediction": page7,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()

