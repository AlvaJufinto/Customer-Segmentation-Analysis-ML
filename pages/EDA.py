import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import read_customer_data

st.set_page_config(layout="wide")

st.title("Exploratory Data Analysis")

df = read_customer_data()

df.columns = df.columns.str.strip()
numeric_cols = df.select_dtypes(include='number').columns.tolist()

# remove id-like columns (optional)
for col in ['client_id']:
    if col in numeric_cols:
        numeric_cols.remove(col)
st.info("""
ℹ️  
This analysis explores customer financial behavior and demographics.

Goals:
- Understand spending patterns  
- Identify high-value customers  
- Discover relationships between income and spending  
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Preview")
    st.dataframe(df.head())

with col2:
    st.subheader("Statistics")
    st.dataframe(df.describe())

st.header("Visual Analysis")

col1, col2 = st.columns(2)

label_map = {
    "total_spend": "Total Spend ($)",
    "avg_spend": "Average Spend ($)",
    "transaction_count": "Number of Transactions",
    "avg_hour": "Average Transaction Hour",
    "current_age": "Age",
    "yearly_income": "Yearly Income ($)",
    "total_debt": "Total Debt ($)",
    "credit_score": "Credit Score",
    "num_credit_cards": "Number of Credit Cards"
}

with col1:
    st.subheader("Histogram")
    hist_col = st.selectbox("Select Column", numeric_cols, key="hist")

    fig, ax = plt.subplots()
    ax.hist(df[hist_col].dropna(), bins=30)

    ax.set_title(f"Distribution of {label_map.get(hist_col, hist_col)}")
    ax.set_xlabel(label_map.get(hist_col, hist_col))
    ax.set_ylabel("Frequency")

    st.pyplot(fig)
    
with col2:
    st.subheader("Box Plot")
    box_col = st.selectbox("Box Column", numeric_cols, key="box")

    fig2, ax2 = plt.subplots()
    ax2.boxplot(df[box_col].dropna(), vert=True)

    ax2.set_title(f"Boxplot of {label_map.get(box_col, box_col)}")
    ax2.set_ylabel(label_map.get(box_col, box_col))

    st.pyplot(fig2)
    
col3, col4 = st.columns(2)

with col3:
    st.subheader("Scatter Plot")
    x_col = st.selectbox("X-axis", numeric_cols, key="x")
    y_col = st.selectbox("Y-axis", numeric_cols, key="y")

    fig3, ax3 = plt.subplots()
    ax3.scatter(df[x_col], df[y_col])
    ax3.set_xlabel(x_col)
    ax3.set_ylabel(y_col)

    st.pyplot(fig3)

# ------------------------------
# Correlation
# ------------------------------
with col4:
    st.subheader("Correlation")

    corr = df[numeric_cols].corr()

    fig4, ax4 = plt.subplots()
    cax = ax4.matshow(corr)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)

    fig4.colorbar(cax)
    st.pyplot(fig4)

st.header("Key Business Insights")

# Top spenders
st.subheader("Top 10 Customers by Spending")
top_spenders = df.nlargest(10, 'total_spend')
st.dataframe(top_spenders[['client_id','total_spend','transaction_count']])

# Category distribution
if 'favorite_category' in df.columns:
    st.subheader("Favorite Category Distribution")
    st.bar_chart(df['favorite_category'].value_counts())

# Spending vs Income
if 'yearly_income' in df.columns and 'total_spend' in df.columns:
    st.subheader("Spending vs Income")

    sample_df = df.sample(min(5000, len(df)))

    fig5, ax5 = plt.subplots()

    ax5.scatter(
        sample_df['yearly_income'],
        sample_df['total_spend'],
        alpha=0.5
    )

    ax5.set_title("Customer Spending vs Income")
    ax5.set_xlabel("Yearly Income ($)")
    ax5.set_ylabel("Total Spend ($)")

    ax5.ticklabel_format(style='plain')

    st.pyplot(fig5)