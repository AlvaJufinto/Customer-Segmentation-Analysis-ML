import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from utils import read_customer_data

st.title("Train Your Model")

st.info("""
Train a KMeans clustering model for customer segmentation.
Adjust parameters and explore how segmentation changes.
""")

df = read_customer_data()

st.subheader("Feature Configuration")

recommended_features = [
		'total_spend',
		'avg_spend',
		'transaction_count',
		'avg_hour',
		'yearly_income',
		'credit_score'
]

use_recommended = st.checkbox("Use Recommended Features", value=True)

if use_recommended:
		features = recommended_features
else:
		features = st.multiselect(
				"Select Features",
				df.columns.tolist(),
				default=recommended_features
		)

st.caption(f"Using features: {features}")

st.subheader("Model Parameters")

col1, col2 = st.columns(2)

with col1:
		k = st.slider("K Range", 2, 7, 3)

with col2:
		max_iter = st.slider("Max Iterations", 100, 500, 300)

random_state = st.number_input("Random State", 0, 100, 42)

st.subheader("Find Optimal K")

find_k = st.checkbox("Analyze K (Elbow + Silhouette)")

best_k = None

def preprocess_features(df, features):
		df_model = df[features].copy()

		for col in df_model.columns:
				if df_model[col].dtype == 'object':
						# OPTION: use grouping OR one-hot
						df_model = pd.get_dummies(df_model, columns=[col])

		return df_model

if find_k:
		X = preprocess_features(df, features).dropna()

		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(X)

		K_range = range(2, 10)

		inertia_values = []
		silhouette_values = []

		for k_val in K_range:
				km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
				labels = km.fit_predict(X_scaled)

				inertia_values.append(km.inertia_)
				silhouette_values.append(silhouette_score(X_scaled, labels))

		best_k = K_range[np.argmax(silhouette_values)]

		st.success(f"Best K (based on Silhouette Score): {best_k}")

		col1, col2 = st.columns(2)

		# ELBOW
		with col1:
				st.subheader("Elbow Method")

				fig1, ax1 = plt.subplots()
				ax1.plot(K_range, inertia_values, marker='o')
				ax1.set_title("Inertia vs K")
				ax1.set_xlabel("K")
				ax1.set_ylabel("Inertia")

				st.pyplot(fig1)

		# SILHOUETTE
		with col2:
				st.subheader("Silhouette Score")

				fig2, ax2 = plt.subplots()
				ax2.plot(K_range, silhouette_values, marker='o')
				ax2.axvline(best_k, linestyle='--')

				ax2.set_title("Silhouette Score vs K")
				ax2.set_xlabel("K")
				ax2.set_ylabel("Score")

				st.pyplot(fig2)

		st.info("""
		- Elbow: look for the "bend" point  
		- Silhouette: higher is better (closer to 1)
		""")

		#if best_k in [3, 5, 7]:
		#		if st.button("Use Recommended K"):
		#				k = best_k
		#				st.success(f"K updated to {k}")

if st.button("Train Model"):

		if len(features) < 2:
				st.error("Select at least 2 features.")
				st.stop()

		X = df[features].dropna()

		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(X)

		kmeans = KMeans(
				n_clusters=k,
				max_iter=max_iter,
				random_state=random_state,
				n_init=10
		)

		labels = kmeans.fit_predict(X_scaled)

		st.subheader("Model Evaluation")

		inertia = kmeans.inertia_
		silhouette = silhouette_score(X_scaled, labels)

		col1, col2 = st.columns(2)

		with col1:
				st.metric("Inertia", f"{inertia:,.2f}")

		with col2:
				st.metric("Silhouette Score", f"{silhouette:.3f}")

		st.subheader("Cluster Visualization")

		pca = PCA(n_components=2)
		X_pca = pca.fit_transform(X_scaled)

		fig, ax = plt.subplots()

		scatter = ax.scatter(
				X_pca[:, 0],
				X_pca[:, 1],
				c=labels,
				cmap='viridis',
				alpha=0.6
		)

		ax.set_title(f"KMeans Clustering (K={k})")
		ax.set_xlabel("PCA 1")
		ax.set_ylabel("PCA 2")

		st.pyplot(fig)

		st.subheader("Cluster Profile")

		df_clustered = X.copy()
		df_clustered['cluster'] = labels

		cluster_summary = df_clustered.groupby('cluster').mean().round(2)

		st.dataframe(cluster_summary, use_container_width=True)


		st.subheader("Download Results")

		df_result = df.copy()
		df_result.loc[X.index, 'cluster'] = labels

		csv = df_result.to_csv(index=False).encode('utf-8')

		st.download_button(
				label="Download Clustered Data",
				data=csv,
				file_name="clustered_customers.csv",
				mime="text/csv"
		)
	
		def save_results_parquet(df_result):
				os.makedirs("data/output", exist_ok=True)
				df_result.to_parquet("data/output/clustered_data.parquet", index=False)
				
		save_results_parquet(df_result)