import streamlit as st
import pandas as pd
import os

st.title("Customer Insights")

def load_data():
		return pd.read_parquet("data/output/clustered_data.parquet")

file_path = "data/output/clustered_data.parquet"

if not os.path.exists(file_path):
		st.error("No clustered data found. Please train the model first.")
		st.stop()

df = load_data()

st.subheader("Sample Data (20 rows)")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("Cluster Overview")

cluster_counts = df['cluster'].value_counts().sort_index()

col1, col2 = st.columns(2)

with col1:
		st.write("Cluster Distribution")
		st.bar_chart(cluster_counts)

with col2:
		st.write("Cluster Count")
		for c, val in cluster_counts.items():
				st.write(f"Cluster {c}: {val} users")

st.subheader("Cluster Details")

cluster_profile = df.groupby('cluster').mean(numeric_only=True).round(2)
st.dataframe(cluster_profile, use_container_width=True)

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

				st.subheader(f"User belongs to Cluster {cluster_id}")
				#avg_spend = user['total_spend'].values[0]
				#avg_tx = user['transaction_count'].values[0]

				#st.subheader("Business Recommendation")

				#if avg_spend > df['total_spend'].quantile(0.66):
				#		st.markdown("""
				#		**High Value Customer**
				#		- Offer premium services
				#		- Give loyalty rewards
				#		- Personalized promotions
				#		""")
				#elif avg_spend > df['total_spend'].quantile(0.33):
				#		st.markdown("""
				#		**Mid Value Customer**
				#		- Upsell products
				#		- Offer discounts to increase engagement
				#		""")
				#else:
				#		st.markdown("""
				#		**Low Value Customer**
				#		- Send promotions
				#		- Re-engagement campaigns
				#		- Discounts / cashback
				#		""")

