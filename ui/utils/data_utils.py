import pandas as pd
import numpy as np
import os
import streamlit as st
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

BASE_DIR = Path(__file__).resolve().parent.parent.parent

@st.cache_data
def load_dataset_for_eda():
    """Load the main dataset for EDA analysis."""
    try:
        csv_path = os.path.join(BASE_DIR, "Data", "address_data_combined.csv")
        st.info(f"Loading data from: {csv_path}")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return df.dropna()
        else:
            st.error(f"CSV file not found at {csv_path}")
            # Try alternative locations
            alt_paths = [
                os.path.join(os.path.dirname(__file__), "..", "..", "Data", "address_data_combined.csv"),
                os.path.join(BASE_DIR, "data", "address_data_combined.csv"),
                os.path.join(os.path.dirname(__file__), "..", "data", "address_data_combined.csv")
            ]
            
            for path in alt_paths:
                st.info(f"Trying alternative path: {path}")
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    st.success(f"Successfully loaded data from: {path}")
                    return df.dropna()
            
            # If all paths failed, return empty dataframe
            raise FileNotFoundError(f"Could not find CSV file in any expected location")
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        # Return empty dataframe with sample columns
        return pd.DataFrame({
            'Address': ['0x...'],
            'FLAG': [0],
            'Sent tnx': [0],
            'Received tnx': [0],
            'total transactions (including tnx to create contract': [0]
        })

def clean_dataset(df):
    """Clean the dataset by removing NaN values and empty columns."""
    df_clean = df.dropna()
    empty_columns = [
        ' ERC20 avg time between contract tnx',
        ' ERC20 max val sent contract',
        ' ERC20 min val sent contract',
        ' ERC20 avg val sent contract',
        ' ERC20 avg time between sent tnx',
        ' ERC20 avg time between rec tnx',
        ' ERC20 avg time between rec 2 tnx'
    ]
    df_clean = df_clean.drop(columns=empty_columns, errors='ignore')
    return df_clean

def get_correlation_analysis(df, threshold=0.7):
    """Perform correlation analysis and return high correlation pairs."""
    num_df = df.select_dtypes(include=np.number)
    corr = num_df.corr().round(2)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sim = corr.where(~mask).stack().reset_index()
    sim.columns = ['feature_1','feature_2','corr']
    high_corr_pairs = sim[sim['corr'].abs() >= threshold].sort_values(by='corr', ascending=False)
    
    return corr, mask, high_corr_pairs

def perform_dimensionality_reduction(df_clean):
    """Perform PCA and t-SNE dimensionality reduction."""
    num_df = df_clean.select_dtypes(include=np.number)
    
    if len(df_clean) <= 10:
        return None, None, None, None
    
    # PCA
    scaled = MinMaxScaler().fit_transform(num_df.drop(columns=['FLAG'], errors='ignore'))
    pca2 = PCA(2).fit_transform(scaled)
    
    # Calculate explained variance
    pca = PCA(2).fit(scaled)
    explained_variance = pca.explained_variance_ratio_ * 100
    
    # t-SNE
    with st.spinner("Computing t-SNE projection (this may take a moment)..."):
        tsne_feats = TSNE(learning_rate=50, random_state=42).fit_transform(pca2)
    
    return pca2, tsne_feats, explained_variance, scaled

def get_transaction_column(df):
    """Find the appropriate transaction count column."""
    if 'Sent tnx' in df.columns:
        return 'Sent tnx'
    elif 'total transactions (including tnx to create contract' in df.columns:
        return 'total transactions (including tnx to create contract'
    else:
        # Find any column that might contain transaction counts
        possible_cols = [col for col in df.columns if 'transaction' in str(col).lower() or 'tnx' in str(col).lower()]
        return possible_cols[0] if possible_cols else 'FLAG'  # Use FLAG as fallback
