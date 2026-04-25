# Customer Behavior Analysis Streamlit App

A Streamlit-based customer analysis project that explores credit-card transaction data, performs feature engineering, and segments customers using K-Means clustering.

## Overview

This project analyzes customer behavior using a financial transaction dataset. It combines transaction history, customer demographics, and merchant category information to build a clean dataset for exploration and clustering.

The app includes:
- Exploratory Data Analysis (EDA)
- Data pipeline and feature explanation
- Interactive K-Means clustering model training
- Cluster results and customer insights

## Project Structure

- `Home.py` - Streamlit landing page and app navigation overview.
- `utils.py` - helper functions for loading data and generating cluster segment labels.
- `pages/Data_Pipeline.py` - explains source datasets and preprocessing steps.
- `pages/EDA.py` - exploratory data analysis with summary statistics and visualizations.
- `pages/Model.py` - feature selection, K-Means training, and cluster evaluation.
- `pages/Result_and_Insights.py` - cluster overview and customer lookup after training.
- `data/customer_data.parquet` - main dataset used by the app.
- `data/output/` - output directory for exported clustered data.
- `requirements.txt` - Python dependencies.

## Data Sources

The app is built around a financial transactions dataset and uses:
- Transaction records
- Customer demographic / financial profile data
- Merchant Category Code (MCC) mappings

These data sources are used to derive customer behavior metrics such as total spend, average spend, transaction count, and favorite merchant category.

## Key Features

- Data preview and descriptive statistics
- Histogram, boxplot, scatter plot, and correlation heatmap
- K-Means clustering with selectable features and parameter tuning
- Elbow method and silhouette score analysis for choosing K
- PCA-based visualization of clusters
- Downloadable clustered dataset output
- Customer search by `client_id` to inspect cluster membership

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run Home.py
```

3. Open the local URL shown in the terminal to explore the app.

## Recommended Workflow

1. Start on `Home.py` for the project overview.
2. Visit `Data Pipeline` to understand the dataset and preprocessing.
3. Use `EDA` to inspect distributions, correlations, and customer spending patterns.
4. Go to `Model` to train clustering models and evaluate K selection.
5. Check `Result and Insights` after training to review clusters and search individual customers.

## Dependencies

Based on the project files, the primary dependencies include:
- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `pyarrow`

## Notes

- The app expects `data/customer_data.parquet` to exist.
- Model outputs are saved to `data/output/clustered_data.parquet`.
- If the clustered output file is missing, `Result and Insights` will prompt to train the model first.

## License

This repository is intended for educational and exploratory use in customer segmentation and behavior analysis.