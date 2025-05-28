import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from model_utils import load_data

def calculate_fraud_detection_score(labels, y_true):
    """Calculate how well clustering identifies fraud patterns."""
    if len(y_true) != len(labels):
        return np.nan
    cluster_fraud_rates = {}
    for cluster_id in set(labels):
        if cluster_id != -1:  # Exclude noise
            mask = labels == cluster_id
            if mask.sum() > 0:
                fraud_rate = y_true[mask].mean()
                cluster_fraud_rates[cluster_id] = fraud_rate
    return max(cluster_fraud_rates.values()) if cluster_fraud_rates else 0.0

def display_unsupervised_comparison_results(results, model_results, X_processed, y_true):
    """Display comprehensive unsupervised comparison results."""
    results_df = pd.DataFrame(results)
    st.subheader("📋 Unsupervised Model Metrics Summary")
    display_df = results_df.copy()
    numeric_cols = ['Silhouette_Score', 'Fraud_Detection_Score', 'ARI_Score']
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
    display_df['Training_Time'] = display_df['Training_Time'].apply(lambda x: f"{x:.4f}s")
    st.dataframe(display_df, use_container_width=True)

def run_supervised_model_comparison(selected_models, supervised_models, test_size, cv_folds, random_state):
    """Run supervised model comparison."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    X, y = load_data(use_synthetic=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    results = []
    prog = st.progress(0)
    for i, name in enumerate(selected_models):
        model = supervised_models[name]
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        preds = model.predict(X_test)
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds),
            'Recall': recall_score(y_test, preds),
            'F1_Score': f1_score(y_test, preds),
            'Training_Time': train_time
        })
        prog.progress((i + 1) / len(selected_models))
    df = pd.DataFrame(results)
    styled = df.style.format({
        'Accuracy': '{:.4f}', 'Precision': '{:.4f}', 'Recall': '{:.4f}',
        'F1_Score': '{:.4f}', 'Training_Time': '{:.3f}s'
    }).background_gradient(subset=['F1_Score'], cmap='Greens')
    st.subheader("📊 Supervised Model Comparison Results")
    st.dataframe(styled, use_container_width=True)

def run_unsupervised_model_comparison(selected_models, all_models, use_pca, n_components, random_state):
    """Run unsupervised model comparison."""
    X, y = load_data(use_synthetic=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_processed = PCA(n_components=n_components, random_state=random_state).fit_transform(X_scaled) if use_pca else X_scaled
    results = []
    prog = st.progress(0)
    for i, model_name in enumerate(selected_models):
        try:
            model = all_models[model_name]
            start = time.time()
            labels = model.fit_predict(X_processed)
            training_time = time.time() - start
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            sil_score = silhouette_score(X_processed, labels) if n_clusters > 1 else np.nan
            fraud_score = calculate_fraud_detection_score(labels, y)
            ari_score = adjusted_rand_score(y, labels) if len(y) == len(labels) else np.nan
            results.append({
                'Model': model_name,
                'N_Clusters': n_clusters,
                'Silhouette_Score': sil_score,
                'Fraud_Detection_Score': fraud_score,
                'ARI_Score': ari_score,
                'Training_Time': training_time
            })
        except Exception as e:
            st.error(f"Error running {model_name}: {str(e)}")
        prog.progress((i + 1) / len(selected_models))
    display_unsupervised_comparison_results(results, {}, X_processed, y)

def run_combined_analysis(sup_name, sup_model, unsup_name, unsup_model):
    """Run combined analysis of supervised and unsupervised approaches."""
    X, y = load_data(use_synthetic=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    sup_model.fit(X_train, y_train)
    sup_pred = sup_model.predict(X_test)
    from sklearn.metrics import accuracy_score, f1_score
    supervised_results = {
        'accuracy': accuracy_score(y_test, sup_pred),
        'f1': f1_score(y_test, sup_pred)
    }
    unsup_labels = unsup_model.fit_predict(X_scaled)
    n_clusters = len(set(unsup_labels)) - (1 if -1 in unsup_labels else 0)
    sil_score = silhouette_score(X_scaled, unsup_labels) if n_clusters > 1 else np.nan
    fraud_score = calculate_fraud_detection_score(unsup_labels, y)
    unsupervised_results = {
        'n_clusters': n_clusters,
        'silhouette': sil_score,
        'fraud_detection': fraud_score
    }
    st.subheader("🔗 Combined Analysis Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{supervised_results['accuracy']:.4f}")
        st.metric("F1-Score", f"{supervised_results['f1']:.4f}")
    with col2:
        st.metric("Clusters", unsupervised_results['n_clusters'])
        st.metric("Silhouette Score", f"{unsupervised_results['silhouette']:.4f}" if not np.isnan(unsupervised_results['silhouette']) else "N/A")
        st.metric("Fraud Detection", f"{unsupervised_results['fraud_detection']:.4f}")
