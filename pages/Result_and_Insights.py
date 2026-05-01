import streamlit as st
import pandas as pd
import os

st.title("Customer Insights Dashboard")

def load_data():
    return pd.read_parquet("data/output/clustered_data.parquet")

file_path = "data/output/clustered_data.parquet"

if not os.path.exists(file_path):
    st.error("No clustered data found. Please train the model first.")
    st.stop()

df = load_data()

df['age'] = 2026 - df['birth_year']
df['spend_per_tx'] = df['total_spend'] / df['transaction_count']
df['debt_ratio'] = df['total_debt'] / df['yearly_income']

business_features = [
    'total_spend',
    'avg_spend',
    'transaction_count',
    'credit_score',
    'total_debt',
    'yearly_income',
    'age',
    'spend_per_tx',
    'debt_ratio'
]

def build_cluster_persona(df):
    profile = df.groupby('cluster').mean(numeric_only=True)

    personas = {}

    for c in profile.index:
        row = profile.loc[c]

        if row['total_spend'] > df['total_spend'].mean() and row['credit_score'] > 700:
            label = "Premium Customers"
        elif row['transaction_count'] > df['transaction_count'].mean():
            label = "Active Users"
        elif row['total_debt'] > df['total_debt'].mean():
            label = "High Risk Customers"
        else:
            label = "Standard Customers"

        personas[c] = label

    return personas

cluster_persona = build_cluster_persona(df)

st.subheader("Sample Data")
st.dataframe(df.head(20), use_container_width=True)
st.subheader("Cluster Overview")

cluster_counts = df['cluster'].value_counts().sort_index()

col1, col2 = st.columns(2)

with col1:
    st.write("Cluster Distribution")
    st.bar_chart(cluster_counts)

with col2:
    st.write("Cluster Summary")

    for c, val in cluster_counts.items():
        st.write(f"""
        **Cluster {c} - {cluster_persona[c]}**
        - Users: {val}
        """)
st.subheader("Cluster Profile (Business Metrics)")

cluster_profile = df.groupby('cluster')[business_features].mean().round(2)

st.dataframe(cluster_profile, use_container_width=True)

# start
st.write("### Daftar Pelanggan Berdasarkan Persona")

df_display = df.copy()
df_display['persona'] = df_display['cluster'].map(cluster_persona)

unique_personas = list(set(cluster_persona.values()))
tabs = st.tabs(unique_personas)

for tab, persona_name in zip(tabs, unique_personas):
    with tab:
        st.write(f"Menampilkan pelanggan dengan persona: **{persona_name}**")
        filtered_data = df_display[df_display['persona'] == persona_name]
        st.dataframe(filtered_data, use_container_width=True)

st.subheader("Cluster Comparison")
metric = st.selectbox("Choose metric", business_features)
# end

st.bar_chart(df.groupby('cluster')[metric].mean())

def get_recommendation(persona):
    if "Premium" in persona:
        return "Offer loyalty program, premium perks, exclusive benefits"
    elif "Active" in persona:
        return "Boost engagement with promotions & bundling"
    elif "High Risk" in persona:
        return "Credit monitoring & risk mitigation strategy"
    else:
        return "Reactivation campaign / retention strategy"

st.subheader("Find Customer")

user_id = st.number_input("Enter Client ID", min_value=0, step=1)

if st.button("Search"):
    user = df[df['client_id'] == user_id]

    if user.empty:
        st.warning("User not found")
    else:
        st.success("User found")
        st.dataframe(user, use_container_width=True)

        cluster_id = int(user['cluster'].values[0])
        persona = cluster_persona[cluster_id]

        st.subheader(f"Cluster {cluster_id} - {persona}")

        cluster_data = df[df['cluster'] == cluster_id]

        # Comparison
        st.write("Comparison vs Cluster Average")

        comparison = pd.DataFrame({
            "User": user[business_features].mean(),
            "Cluster Avg": cluster_data[business_features].mean()
        })

        st.dataframe(comparison)

        # Recommendation
        st.subheader("Business Recommendation")
        st.info(get_recommendation(persona))