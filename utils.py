import pandas as pd
import streamlit as st

@st.cache_data
def load_rfm():
    return pd.read_parquet('data/rfm_table.parquet')

@st.cache_data
def load_category():
    return pd.read_parquet('data/agg_category.parquet')

@st.cache_data
def load_hour():
    return pd.read_parquet('data/agg_hour.parquet')