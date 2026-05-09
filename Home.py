import streamlit as st

st.set_page_config(page_title="Customer Behavior Analysis", layout="wide")

st.title("Customer Behavior Analysis based on Credit Card Transaction History")

st.markdown("""
## Group 8
1. Abraham Gregorius Anderson Thio - 2802473504
2. Alwan Athallah Mumtaz - 2802473896
3. Axel Sanjiro Yang - 2802472400
4. Sean Spencer - 2802466953
5. Stanislaus Alva Jufinto - 2802473214     
            
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