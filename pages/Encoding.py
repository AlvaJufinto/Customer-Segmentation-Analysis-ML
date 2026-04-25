import streamlit as st
import pandas as pd

st.title("Data Encoding")

st.info("ℹ️ This section explains how each feature is encoded before modeling.")
#
#st.subheader("Customer Segmentation (Output)")

#k = st.slider("Select Number of Segments (K)", 2, 10, 3)

def generate_segments(k):
    base_labels = [
        "Top Customer",
        "Loyal",
        "Potential",
        "At Risk",
        "Hibernating",
        "Low Value",
        "New",
        "Big Spenders",
        "Frequent Buyers",
        "Occasional"
    ]

    descriptions = [
        "Nasabah Prioritas",
        "Consistent and loyal behavior",
        "Showing growth potential",
        "Declining engagement",
        "Inactive or sleeping customers",
        "Very low activity",
        "Recently acquired customers",
        "High spending behavior",
        "Frequent transaction behavior",
        "Irregular purchase pattern"
    ]

    result = []
    for i in range(k):
        label = base_labels[i] if i < len(base_labels) else f"Segment {i+1}"
        desc = descriptions[i] if i < len(descriptions) else "Customer group"
        result.append((label, desc))

    return result
  
#segments = generate_segments(k)

#target_df = pd.DataFrame(segments, columns=["Label", "Description"])

#st.subheader("Customer Segmentation (Output)")
#st.dataframe(target_df, use_container_width=True)

st.subheader("Numerical Features")

num_df = pd.DataFrame({
    "Feature": [
        "total_spend",
        "avg_spend",
        "transaction_count",
        "avg_hour",
        "current_age",
        "yearly_income",
        "total_debt",
        "credit_score",
        "num_credit_cards"
    ],
    "Encoding": ["StandardScaler"] * 9,
    "Description": [
        "Total customer spending",
        "Average transaction value",
        "Number of transactions",
        "Average transaction time",
        "Customer age",
        "Annual income",
        "Total debt",
        "Credit score",
        "Number of credit cards"
    ]
})

st.dataframe(num_df, use_container_width=True)

st.subheader("Categorical Features")

cat_df = pd.DataFrame({
    "Feature": ["gender", "favorite_category"],
    "Encoding": ["Label Encoding", "One-Hot Encoding"],
    "Description": [
        "Customer gender (Male/Female)",
        "Most frequent spending category"
    ]
})

st.dataframe(cat_df, use_container_width=True)

st.subheader("Gender Encoding")

gender_df = pd.DataFrame({
    "Label": [0, 1],
    "Description": ["Female", "Male"]
})

st.dataframe(gender_df, use_container_width=True)

st.subheader("Favorite Category Encoding")

cat_example_df = pd.DataFrame({
    "Category": [
        "Eating Places and Restaurants",
        "Grocery Stores",
        "Service Stations",
        "Drinking Places"
    ],
    "Encoded As": [
        "[1,0,0,0]",
        "[0,1,0,0]",
        "[0,0,1,0]",
        "[0,0,0,1]"
    ]
})

st.dataframe(cat_example_df, use_container_width=True)

st.subheader("Dropped Features")

drop_df = pd.DataFrame({
    "Feature": ["client_id", "address", "birth_year", "birth_month"],
    "Reason": [
        "Identifier only",
        "Too granular / high cardinality",
        "Redundant with age",
        "Low predictive value"
    ]
})

st.dataframe(drop_df, use_container_width=True)