import pandas as pd
import streamlit as st

@st.cache_data
def read_customer_data():
    return pd.read_parquet('data/customer_data.parquet')
  
    """
    Generate segment labels based on cluster ranking.

    Parameters:
    - df_clustered: DataFrame with 'cluster', 'total_spend', 'transaction_count'
    - k: number of clusters

    Returns:
    - dict: cluster_id -> {name, description}
    """

    # Step 1: ranking cluster (multi-metric biar ga bias)
    cluster_rank = (
        df_clustered
        .groupby('cluster')
        .agg({
            'total_spend': 'mean',
            'transaction_count': 'mean'
        })
    )

    # normalize biar comparable
    cluster_rank['spend_norm'] = (
        cluster_rank['total_spend'] - cluster_rank['total_spend'].min()
    ) / (cluster_rank['total_spend'].max() - cluster_rank['total_spend'].min() + 1e-9)

    cluster_rank['tx_norm'] = (
        cluster_rank['transaction_count'] - cluster_rank['transaction_count'].min()
    ) / (cluster_rank['transaction_count'].max() - cluster_rank['transaction_count'].min() + 1e-9)

    # combine score
    cluster_rank['score'] = (
        0.6 * cluster_rank['spend_norm'] +
        0.4 * cluster_rank['tx_norm']
    )

    # sort highest → lowest
    cluster_rank = cluster_rank.sort_values(by='score', ascending=False)

    cluster_order = cluster_rank.index.tolist()

    # label pool
    labels_pool = [
        ("Champion", "Top customers, high value"),
        ("Loyal", "Consistent and valuable"),
        ("Big Spenders", "High spend but less frequent"),
        ("Potential", "Growing customers"),
        ("Average", "Moderate behavior"),
        ("At Risk", "Declining engagement"),
        ("Low Value", "Low activity customers"),
    ]

    # fallback kalau k > pool
    if k > len(labels_pool):
        labels_pool += [(f"Segment {i}", "Customer group") for i in range(len(labels_pool), k)]

    labels = labels_pool[:k]

    # mapping
    cluster_map = {}

    for i, cluster_id in enumerate(cluster_order):
        name, desc = labels[i]
        cluster_map[cluster_id] = {
            "name": name,
            "description": desc
        }

    return cluster_map