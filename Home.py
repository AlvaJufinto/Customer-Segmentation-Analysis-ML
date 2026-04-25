import streamlit as st

st.set_page_config(page_title="Customer Behavior Analysis", layout="wide")

st.title("Customer Behavior Analysis based on Credit Card Transaction History")

st.markdown("""
## Overview

This project analyzes customer behavior based on transaction data and segments customers using RFM and K-Means Clustering.

---

## Objectives
- Understand customer transaction patterns
- Identify customer segments
- Provide actionable business insights

---

## Dataset
The dataset includes:
- customer transactions
- merchant categories (MCC)
- timestamps

---

## Methodology
1. Data Cleaning & Feature Engineering
2. K-Means Clustering  
3. Business Insights  

---

## Navigation
Use the sidebar to explore:
- EDA: explore the data  
- Preprocessing: data preparation steps  
- Clustering ML model  
- Insights: business recommendations  
""")