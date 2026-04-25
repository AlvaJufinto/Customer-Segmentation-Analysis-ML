import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.title("Data Description")

st.info("""
ℹ️  
This project uses a financial dataset from:
https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets/data

We use three datasets:

- **Transactions Data**: records all customer transactions (spending behavior)  
- **Users Data**: contains customer demographic and financial profile  
- **MCC Codes**: maps transaction codes to spending categories  

These datasets allow us to analyze **who the customer is, what they spend on, and how they behave**.
""")
dataset_option = st.selectbox(
  
    "Select Dataset",
    ["Transactions Data", "Users Data", "MCC Codes"]
)

if dataset_option == "Transactions Data":
    st.header("Transactions Dataset")

    tx_dummy = pd.DataFrame({
        "id": [7475327, 7475328],
        "date": ["2010-01-01 00:01:00", "2010-01-01 00:02:00"],
        "client_id": [1556, 561],
        "card_id": [2972, 4575],
        "amount": [-77.00, 14.57],
        "use_chip": ["Swipe", "Swipe"],
        "merchant_id": [59935, 67570],
        "merchant_city": ["Beulah", "Bettendorf"],
        "merchant_state": ["ND", "IA"],
        "mcc": [5499, 5311]
    })

    st.dataframe(tx_dummy)

    st.subheader("Column Explanation")

    st.markdown("""
- **id**:  unique transaction ID  
- **date**:  transaction timestamp  
- **client_id**:  user ID (will link to users_data)  
- **card_id**:  card used  
- **amount**:  transaction value (negative = refund)  
- **use_chip**:  transaction type (online/swipe)  
- **merchant_id**:  merchant identifier  
- **merchant_city/state**:  merchant location  
- **mcc**:  Merchant Category Code  
    """)

    st.info("This is raw transaction data: used to compute spending behavior")

# ==============================
# USERS (DUMMY)
# ==============================
elif dataset_option == "Users Data":
    st.header("Users Dataset")

    users_dummy = pd.DataFrame({
        "id": [825, 1746],
        "current_age": [53, 53],
        "gender": ["Female", "Female"],
        "yearly_income": [59696, 77254],
        "total_debt": [127613, 191349],
        "credit_score": [787, 701],
        "num_credit_cards": [5, 5]
    })

    st.dataframe(users_dummy)

    st.subheader("Column Explanation")

    st.markdown("""
- **id**: user identifier (same as client_id)  
- **current_age**: customer age  
- **gender**: demographic info  
- **yearly_income**: income  
- **total_debt**: financial liability  
- **credit_score**: creditworthiness  
- **num_credit_cards**: number of cards  
    """)

    st.info("Adds demographic & financial profile to each customer")

# ==============================
# MCC (DUMMY)
# ==============================
elif dataset_option == "MCC Codes":
    st.header("MCC Codes")

    mcc_dummy = pd.DataFrame({
        "mcc": [5812, 5411, 5541],
        "category": [
            "Eating Places and Restaurants",
            "Grocery Stores",
            "Gas Stations"
        ]
    })

    st.dataframe(mcc_dummy)

    st.subheader("Explanation")

    st.markdown("""
- **mcc**: Merchant Category Code  
- **category**: business type  

Used to transform raw transactions into meaningful categories.
    """)

# ==============================
# PIPELINE EXPLANATION
# ==============================
st.header("How Data is Processed")

st.markdown("""
### Step 1 (Raw Data)
We start with:
- Transactions: customer activity  
- Users: demographics  
- MCC: transaction category  

---

### Step 2 (Feature Engineering)
We transform:
- amount: total spending  
- date: hour, day, month  
- mcc: category  

---

### Step 3 (Aggregation)
We group transactions by user:
- total_spend  
- avg_spend  
- transaction_count  
- favorite_category  

---

### Step 4 (Merge)
We combine:
- transaction features  
- user demographics  

---

### Final Output
A clean dataset with:
- customer behavior  
- financial profile  
- ready for ML clustering
""")

st.header("Final Dataset Example")

final_dummy = pd.DataFrame({
    "client_id": [0, 1],
    "total_spend": [780919.67, 367921.37],
    "avg_spend": [61.03, 36.52],
    "transaction_count": [12795, 10073],
    "favorite_category": ["Drinking Places", "Taxi"],
    "current_age": [33, 43],
    "yearly_income": [59613, 45360],
    "credit_score": [763, 704]
})

st.dataframe(final_dummy)