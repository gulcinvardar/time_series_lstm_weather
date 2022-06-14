import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px



def page7():
    st.subheader("Prediction")
    st.title("  ")
    st.title("  ")

    col1, col2, col3 = st.columns([1,4,1])
    with col2:
        st.image('pictures/timestep_5_compare.jpg',
                        width=700)
    
    
    'Tomorrow weather'