import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import warnings
warnings.filterwarnings('ignore')

from model_utils import load_data, get_unsupervised_models, get_supervised_models

def render_unified_analysis_tab():
    """Render unified analysis combining supervised and unsupervised approaches."""
    st.header("🔬 Unified Analysis")
    
    st.markdown("""
    ### Comprehensive Machine Learning Analysis
    
    This unified analysis combines supervised and unsupervised machine learning approaches
    to provide comprehensive insights into fraud detection patterns and model performance.
    """)
    
    # Main analysis tabs
    analysis_tabs = st.tabs([
        "🎯 Unsupervised ML Analysis",
        "🤖 Supervised vs Unsupervised",
        "📊 Pattern Discovery",
        "🔍 Anomaly Detection Suite",
        "📈 Cluster-Based Insights"
    ])
    
    with analysis_tabs[0]:
        _render_unsupervised_ml_analysis()
    
    with analysis_tabs[1]:
        _render_supervised_vs_unsupervised()
    
    with analysis_tabs[2]:
        _render_pattern_discovery()
    
    with analysis_tabs[3]:
        _render_anomaly_detection_suite()
    
    with analysis_tabs[4]:
        _render_cluster_based_insights()

def _render_unsupervised_ml_analysis():
    """Render comprehensive unsupervised ML analysis."""
    st.subheader("🎯 Unsupervised Machine Learning Analysis")
    
    st.markdown("""
    ### Advanced Clustering & Dimensionality Reduction
    
    Explore data structure using various unsupervised learning techniques
    to discover hidden patterns and fraud indicators.
    """)
    
    # Unsupervised ML subtabs
    unsup_tabs = st.tabs([
        "Clustering Analysis",
        "Dimensionality Reduction", 
        "Anomaly Detection",
        "Association Analysis",
        "Model Comparison"
    ])
    
    with unsup_tabs[0]:
        _render_clustering_analysis()
    
    with unsup_tabs[1]:
        _render_dimensionality_reduction()
    
    with unsup_tabs[2]:
        _render_unsupervised_anomaly_detection()
    
    with unsup_tabs[3]:
        _render_association_analysis()
    
    with unsup_tabs[4]:
        _render_unsupervised_model_comparison()

def _render_clustering_analysis():
    """Render comprehensive clustering analysis."""
    st.subheader("Clustering Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Clustering algorithm selection
        clustering_methods = {
            "K-Means": KMeans(n_clusters=3, random_state=42),
            "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
            "Agglomerative": AgglomerativeClustering(n_clusters=3),
            "Gaussian Mixture": GaussianMixture(n_components=3, random_state=42)
        }
        
        selected_methods = st.multiselect(
            "Select clustering methods",
            options=list(clustering_methods.keys()),
            default=["K-Means", "DBSCAN"],
            key="clustering_methods"
        )
        
        # Parameter configuration
        if "K-Means" in selected_methods:
            n_clusters_kmeans = st.slider("K-Means clusters", 2, 10, 3, key="kmeans_clusters")
            clustering_methods["K-Means"] = KMeans(n_clusters=n_clusters_kmeans, random_state=42)
        
        if "DBSCAN" in selected_methods:
            eps = st.slider("DBSCAN epsilon", 0.1, 2.0, 0.5, 0.1, key="dbscan_eps")
            min_samples = st.slider("DBSCAN min samples", 2, 20, 5, key="dbscan_min")
            clustering_methods["DBSCAN"] = DBSCAN(eps=eps, min_samples=min_samples)
        
        if "Agglomerative" in selected_methods:
            n_clusters_agg = st.slider("Agglomerative clusters", 2, 10, 3, key="agg_clusters")
            clustering_methods["Agglomerative"] = AgglomerativeClustering(n_clusters=n_clusters_agg)
        
        if "Gaussian Mixture" in selected_methods:
            n_components_gmm = st.slider("GMM components", 2, 10, 3, key="gmm_components")
            clustering_methods["Gaussian Mixture"] = GaussianMixture(n_components=n_components_gmm, random_state=42)
        
        if st.button("Run Clustering Analysis", key="clustering_btn"):
            if selected_methods:
                X, y = load_data(use_synthetic=True)
                
                # Preprocess data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Apply PCA for visualization
                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(X_scaled)
                
                # Run clustering algorithms
                clustering_results = {}
                clustering_metrics = {}
                
                for method_name in selected_methods:
                    method = clustering_methods[method_name]
                    
                    try:
                        if method_name == "Gaussian Mixture":
                            labels = method.fit_predict(X_scaled)
                        else:
                            labels = method.fit_predict(X_scaled)
                        
                        clustering_results[method_name] = labels
                        
                        # Calculate metrics
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        
                        if n_clusters > 1:
                            silhouette = silhouette_score(X_scaled, labels)
                            ari = adjusted_rand_score(y, labels)
                            nmi = normalized_mutual_info_score(y, labels)
                        else:
                            silhouette = ari = nmi = 0
                        
                        # Fraud detection effectiveness
                        fraud_rates = []
                        for cluster_id in set(labels):
                            if cluster_id != -1:
                                mask = labels == cluster_id
                                if mask.sum() > 0:
                                    fraud_rate = y[mask].mean()
                                    fraud_rates.append(fraud_rate)
                        
                        max_fraud_rate = max(fraud_rates) if fraud_rates else 0
                        
                        clustering_metrics[method_name] = {
                            'n_clusters': n_clusters,
                            'silhouette': silhouette,
                            'ari': ari,
                            'nmi': nmi,
                            'max_fraud_rate': max_fraud_rate,
                            'noise_ratio': (labels == -1).sum() / len(labels) if -1 in labels else 0
                        }
                        
                    except Exception as e:
                        st.warning(f"Error with {method_name}: {str(e)}")
                
                # Visualization
                n_methods = len(clustering_results)
                if n_methods > 0:
                    fig, axes = plt.subplots(2, min(2, n_methods), figsize=(15, 10))
                    if n_methods == 1:
                        axes = axes.reshape(2, 1)
                    elif n_methods == 2:
                        axes = axes.reshape(2, 2)
                    else:
                        fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=(20, 10))
                    
                    for idx, (method_name, labels) in enumerate(clustering_results.items()):
                        row = idx // 2
                        col = idx % 2
                        
                        if n_methods <= 2:
                            ax = axes[row, col] if n_methods > 1 else axes[row]
                        else:
                            ax = axes[row, col]
                        
                        # Plot clusters
                        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.6)
                        ax.set_title(f'{method_name} Clustering')
                        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
                        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
                    
                    # Hide unused subplots
                    for idx in range(len(clustering_results), axes.size):
                        row = idx // 2
                        col = idx % 2
                        if n_methods <= 2:
                            ax = axes[row, col] if n_methods > 1 else axes[row]
                        else:
                            ax = axes[row, col]
                        ax.set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Metrics comparison
                    st.subheader("Clustering Performance Metrics")
                    metrics_df = pd.DataFrame(clustering_metrics).T
                    metrics_df = metrics_df.round(4)
                    
                    # Style the dataframe
                    styled_df = metrics_df.style.background_gradient(subset=['silhouette', 'ari', 'nmi', 'max_fraud_rate'], cmap='Greens')
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Best method recommendation
                    if len(clustering_metrics) > 1:
                        # Composite score (higher is better)
                        composite_scores = {}
                        for method, metrics in clustering_metrics.items():
                            score = (metrics['silhouette'] * 0.3 + 
                                   metrics['ari'] * 0.3 + 
                                   metrics['max_fraud_rate'] * 0.4)
                            composite_scores[method] = score
                        
                        best_method = max(composite_scores.keys(), key=lambda k: composite_scores[k])
                        st.success(f"🏆 **Recommended method:** {best_method} (composite score: {composite_scores[best_method]:.3f})")
    
    with col2:
        st.markdown("### Clustering Guidelines")
        st.info("""
        **Method Selection:**
        
        **K-Means:**
        - Good for spherical clusters
        - Requires pre-specifying K
        - Fast and scalable
        
        **DBSCAN:**
        - Handles noise and outliers
        - Finds arbitrary shaped clusters
        - Parameter sensitive
        
        **Agglomerative:**
        - Hierarchical clustering
        - Good for small datasets
        - Can specify cluster count
        
        **Gaussian Mixture:**
        - Probabilistic clustering
        - Handles overlapping clusters
        - Provides uncertainty estimates
        """)

def _render_dimensionality_reduction():
    """Render dimensionality reduction analysis."""
    st.subheader("Dimensionality Reduction")
    
    st.markdown("""
    ### Data Visualization in Lower Dimensions
    
    Apply various dimensionality reduction techniques to visualize high-dimensional
    fraud detection data and discover hidden patterns.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        reduction_methods = {
            "PCA": PCA(n_components=2, random_state=42),
            "t-SNE": TSNE(n_components=2, random_state=42, perplexity=30),
            "PCA (3D)": PCA(n_components=3, random_state=42)
        }
        
        selected_method = st.selectbox(
            "Select dimensionality reduction method",
            options=list(reduction_methods.keys()),
            key="dim_reduction_method"
        )
        
        # Method-specific parameters
        if selected_method == "t-SNE":
            perplexity = st.slider("t-SNE perplexity", 5, 50, 30, key="tsne_perplexity")
            reduction_methods["t-SNE"] = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        
        color_by = st.selectbox(
            "Color points by",
            options=["Fraud Labels", "Feature Value", "Cluster Labels"],
            key="color_by"
        )
        
        if st.button("Apply Dimensionality Reduction", key="dim_reduction_btn"):
            X, y = load_data(use_synthetic=True)
            
            # Preprocess data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply selected method
            method = reduction_methods[selected_method]
            
            with st.spinner(f"Applying {selected_method}..."):
                if selected_method == "PCA (3D)":
                    X_reduced = method.fit_transform(X_scaled)
                    
                    # 3D visualization
                    fig = plt.figure(figsize=(12, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    if color_by == "Fraud Labels":
                        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], 
                                           c=y, cmap='RdYlBu', alpha=0.6)
                        plt.colorbar(scatter, ax=ax, label='Fraud (1=Fraud, 0=Legitimate)')
                    elif color_by == "Feature Value":
                        feature_idx = st.selectbox("Select feature", range(X.shape[1]), key="feature_3d")
                        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], 
                                           c=X[:, feature_idx], cmap='viridis', alpha=0.6)
                        plt.colorbar(scatter, ax=ax, label=f'Feature {feature_idx}')
                    else:  # Cluster Labels
                        kmeans = KMeans(n_clusters=3, random_state=42)
                        cluster_labels = kmeans.fit_predict(X_scaled)
                        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], 
                                           c=cluster_labels, cmap='tab10', alpha=0.6)
                        plt.colorbar(scatter, ax=ax, label='Cluster')
                    
                    ax.set_xlabel(f'PC1 ({method.explained_variance_ratio_[0]:.2%})')
                    ax.set_ylabel(f'PC2 ({method.explained_variance_ratio_[1]:.2%})')
                    ax.set_zlabel(f'PC3 ({method.explained_variance_ratio_[2]:.2%})')
                    ax.set_title(f'3D PCA Visualization')
                    
                else:
                    X_reduced = method.fit_transform(X_scaled)
                    
                    # 2D visualization
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    if color_by == "Fraud Labels":
                        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='RdYlBu', alpha=0.6)
                        plt.colorbar(scatter, ax=ax, label='Fraud (1=Fraud, 0=Legitimate)')
                    elif color_by == "Feature Value":
                        feature_idx = st.selectbox("Select feature", range(X.shape[1]), key="feature_2d")
                        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                           c=X[:, feature_idx], cmap='viridis', alpha=0.6)
                        plt.colorbar(scatter, ax=ax, label=f'Feature {feature_idx}')
                    else:  # Cluster Labels
                        kmeans = KMeans(n_clusters=3, random_state=42)
                        cluster_labels = kmeans.fit_predict(X_scaled)
                        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                           c=cluster_labels, cmap='tab10', alpha=0.6)
                        plt.colorbar(scatter, ax=ax, label='Cluster')
                    
                    if selected_method == "PCA":
                        ax.set_xlabel(f'PC1 ({method.explained_variance_ratio_[0]:.2%})')
                        ax.set_ylabel(f'PC2 ({method.explained_variance_ratio_[1]:.2%})')
                    else:
                        ax.set_xlabel('Dimension 1')
                        ax.set_ylabel('Dimension 2')
                    
                    ax.set_title(f'{selected_method} Visualization')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Explained variance for PCA
                if "PCA" in selected_method:
                    st.subheader("Principal Component Analysis Results")
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("PC1 Variance", f"{method.explained_variance_ratio_[0]:.2%}")
                    with col_b:
                        st.metric("PC2 Variance", f"{method.explained_variance_ratio_[1]:.2%}")
                    with col_c:
                        if len(method.explained_variance_ratio_) > 2:
                            st.metric("PC3 Variance", f"{method.explained_variance_ratio_[2]:.2%}")
                        
                        total_var = method.explained_variance_ratio_[:2].sum()
                        st.metric("Total Variance (2D)", f"{total_var:.2%}")
    
    with col2:
        st.markdown("### Reduction Methods")
        st.info("""
        **PCA (Principal Component Analysis):**
        - Linear dimensionality reduction
        - Preserves global structure
        - Fast and interpretable
        
        **t-SNE (t-Distributed Stochastic Neighbor Embedding):**
        - Non-linear dimensionality reduction
        - Good for visualization
        - Preserves local structure
        
        **3D PCA:**
        - Three-dimensional visualization
        - Better data exploration
        - Interactive viewing possible
        """)

def _render_unsupervised_anomaly_detection():
    """Render unsupervised anomaly detection analysis."""
    st.subheader("Unsupervised Anomaly Detection")
    
    st.markdown("""
    ### Multiple Anomaly Detection Approaches
    
    Apply various unsupervised anomaly detection methods to identify
    potential fraud cases without using labeled data.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        anomaly_methods = {
            "Isolation Forest": IsolationForest(contamination=0.1, random_state=42),
            "One-Class SVM": OneClassSVM(nu=0.1),
            "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=0.1),
            "Elliptic Envelope": EllipticEnvelope(contamination=0.1, random_state=42)
        }
        
        selected_anomaly_methods = st.multiselect(
            "Select anomaly detection methods",
            options=list(anomaly_methods.keys()),
            default=["Isolation Forest", "One-Class SVM"],
            key="ua_anomaly_methods"
        )
        
        contamination_rate = st.slider(
            "Expected contamination rate", 
            0.01, 0.5, 0.1, 0.01,
            key="ua_contamination_rate"
        )
        
        # Update contamination rates
        for method_name in selected_anomaly_methods:
            if method_name == "Isolation Forest":
                anomaly_methods[method_name] = IsolationForest(contamination=contamination_rate, random_state=42)
            elif method_name == "One-Class SVM":
                anomaly_methods[method_name] = OneClassSVM(nu=contamination_rate)
            elif method_name == "Local Outlier Factor":
                anomaly_methods[method_name] = LocalOutlierFactor(n_neighbors=20, contamination=contamination_rate)
            elif method_name == "Elliptic Envelope":
                anomaly_methods[method_name] = EllipticEnvelope(contamination=contamination_rate, random_state=42)
        
        if st.button("Run Anomaly Detection", key="ua_anomaly_btn"):
            if selected_anomaly_methods:
                X, y = load_data(use_synthetic=True)
                
                # Preprocess data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Apply PCA for visualization
                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(X_scaled)
                
                anomaly_results = {}
                anomaly_metrics = {}
                
                for method_name in selected_anomaly_methods:
                    method = anomaly_methods[method_name]
                    
                    try:
                        if method_name == "Local Outlier Factor":
                            predictions = method.fit_predict(X_scaled)
                        else:
                            predictions = method.fit(X_scaled).predict(X_scaled)
                        
                        # Convert to binary (1 = normal, -1 = anomaly)
                        anomalies = predictions == -1
                        anomaly_results[method_name] = anomalies
                        
                        # Calculate metrics
                        n_anomalies = anomalies.sum()
                        anomaly_fraud_rate = y[anomalies].mean() if n_anomalies > 0 else 0
                        precision = anomaly_fraud_rate
                        recall = (y[anomalies].sum() / y.sum()) if y.sum() > 0 else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        
                        anomaly_metrics[method_name] = {
                            'n_anomalies': n_anomalies,
                            'anomaly_rate': n_anomalies / len(X),
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'fraud_detection_rate': anomaly_fraud_rate
                        }
                        
                    except Exception as e:
                        st.warning(f"Error with {method_name}: {str(e)}")
                
                # Visualization
                n_methods = len(anomaly_results)
                if n_methods > 0:
                    fig, axes = plt.subplots(2, min(2, n_methods), figsize=(15, 10))
                    if n_methods == 1:
                        axes = [axes] if n_methods == 1 else axes.flatten()
                    else:
                        axes = axes.flatten()
                    
                    for idx, (method_name, anomalies) in enumerate(anomaly_results.items()):
                        ax = axes[idx] if n_methods > 1 else axes[0]
                        
                        # Plot normal and anomalous points
                        normal_mask = ~anomalies
                        ax.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], 
                                 c='blue', alpha=0.6, label='Normal', s=20)
                        ax.scatter(X_pca[anomalies, 0], X_pca[anomalies, 1], 
                                 c='red', alpha=0.8, label='Anomaly', s=30)
                        
                        ax.set_title(f'{method_name}\n({anomalies.sum()} anomalies)')
                        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
                        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                    
                    # Hide unused subplots
                    for idx in range(len(anomaly_results), len(axes)):
                        axes[idx].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Metrics comparison
                    st.subheader("Anomaly Detection Performance")
                    metrics_df = pd.DataFrame(anomaly_metrics).T
                    metrics_df = metrics_df.round(4)
                    
                    # Style the dataframe
                    styled_df = metrics_df.style.background_gradient(subset=['precision', 'recall', 'f1_score'], cmap='Reds')
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Consensus anomalies (detected by multiple methods)
                    if len(anomaly_results) > 1:
                        consensus_anomalies = np.zeros(len(X), dtype=bool)
                        for anomalies in anomaly_results.values():
                            consensus_anomalies |= anomalies
                        
                        # Count how many methods agree
                        agreement_count = np.zeros(len(X))
                        for anomalies in anomaly_results.values():
                            agreement_count += anomalies.astype(int)
                        
                        high_consensus = agreement_count >= len(anomaly_results) // 2 + 1
                        
                        st.subheader("Consensus Analysis")
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric("Total Unique Anomalies", consensus_anomalies.sum())
                        with col_b:
                            st.metric("High Consensus Anomalies", high_consensus.sum())
                        with col_c:
                            if high_consensus.sum() > 0:
                                consensus_fraud_rate = y[high_consensus].mean()
                                st.metric("Consensus Fraud Rate", f"{consensus_fraud_rate:.2%}")
    
    with col2:
        st.markdown("### Anomaly Detection Methods")
        st.info("""
        **Isolation Forest:**
        - Tree-based isolation
        - Good for high dimensions
        - Fast and scalable
        
        **One-Class SVM:**
        - Support vector approach
        - Effective boundary detection
        - Good for complex patterns
        
        **Local Outlier Factor:**
        - Density-based detection
        - Considers local neighborhoods
        - Good for varying densities
        
        **Elliptic Envelope:**
        - Gaussian assumption
        - Covariance-based detection
        - Fast for normal distributions
        """)

def _render_anomaly_detection_suite():
    """Alias for anomaly‐detection tab in unified analysis."""
    st.subheader("Anomaly Detection Suite")
    
    st.markdown("""
    ### Comprehensive Anomaly Detection Analysis
    
    Apply multiple anomaly detection methods to identify potential fraud cases
    using various unsupervised approaches.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        anomaly_methods = {
            "Isolation Forest": IsolationForest(contamination=0.1, random_state=42),
            "One-Class SVM": OneClassSVM(nu=0.1),
            "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=0.1),
            "Elliptic Envelope": EllipticEnvelope(contamination=0.1, random_state=42)
        }
        
        selected_anomaly_methods = st.multiselect(
            "Select anomaly detection methods",
            options=list(anomaly_methods.keys()),
            default=["Isolation Forest", "One-Class SVM"],
            key="suite_anomaly_methods"
        )
        
        contamination_rate = st.slider(
            "Expected contamination rate", 
            0.01, 0.5, 0.1, 0.01,
            key="suite_contamination_rate"
        )
        
        # Update contamination rates
        for method_name in selected_anomaly_methods:
            if method_name == "Isolation Forest":
                anomaly_methods[method_name] = IsolationForest(contamination=contamination_rate, random_state=42)
            elif method_name == "One-Class SVM":
                anomaly_methods[method_name] = OneClassSVM(nu=contamination_rate)
            elif method_name == "Local Outlier Factor":
                anomaly_methods[method_name] = LocalOutlierFactor(n_neighbors=20, contamination=contamination_rate)
            elif method_name == "Elliptic Envelope":
                anomaly_methods[method_name] = EllipticEnvelope(contamination=contamination_rate, random_state=42)
        
        if st.button("Run Anomaly Detection Suite", key="suite_anomaly_btn"):
            if selected_anomaly_methods:
                X, y = load_data(use_synthetic=True)
                
                # Preprocess data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Apply PCA for visualization
                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(X_scaled)
                
                anomaly_results = {}
                anomaly_metrics = {}
                
                for method_name in selected_anomaly_methods:
                    method = anomaly_methods[method_name]
                    
                    try:
                        if method_name == "Local Outlier Factor":
                            predictions = method.fit_predict(X_scaled)
                        else:
                            predictions = method.fit(X_scaled).predict(X_scaled)
                        
                        # Convert to binary (1 = normal, -1 = anomaly)
                        anomalies = predictions == -1
                        anomaly_results[method_name] = anomalies
                        
                        # Calculate metrics
                        n_anomalies = anomalies.sum()
                        anomaly_fraud_rate = y[anomalies].mean() if n_anomalies > 0 else 0
                        precision = anomaly_fraud_rate
                        recall = (y[anomalies].sum() / y.sum()) if y.sum() > 0 else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        
                        anomaly_metrics[method_name] = {
                            'n_anomalies': n_anomalies,
                            'anomaly_rate': n_anomalies / len(X),
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'fraud_detection_rate': anomaly_fraud_rate
                        }
                        
                    except Exception as e:
                        st.warning(f"Error with {method_name}: {str(e)}")
                
                # Visualization
                n_methods = len(anomaly_results)
                if n_methods > 0:
                    fig, axes = plt.subplots(2, min(2, n_methods), figsize=(15, 10))
                    if n_methods == 1:
                        axes = [axes] if n_methods == 1 else axes.flatten()
                    else:
                        axes = axes.flatten()
                    
                    for idx, (method_name, anomalies) in enumerate(anomaly_results.items()):
                        ax = axes[idx] if n_methods > 1 else axes[0]
                        
                        # Plot normal and anomalous points
                        normal_mask = ~anomalies
                        ax.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], 
                                 c='blue', alpha=0.6, label='Normal', s=20)
                        ax.scatter(X_pca[anomalies, 0], X_pca[anomalies, 1], 
                                 c='red', alpha=0.8, label='Anomaly', s=30)
                        
                        ax.set_title(f'{method_name}\n({anomalies.sum()} anomalies)')
                        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
                        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                    
                    # Hide unused subplots
                    for idx in range(len(anomaly_results), len(axes)):
                        axes[idx].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Metrics comparison
                    st.subheader("Anomaly Detection Suite Performance")
                    metrics_df = pd.DataFrame(anomaly_metrics).T
                    metrics_df = metrics_df.round(4)
                    
                    # Style the dataframe
                    styled_df = metrics_df.style.background_gradient(subset=['precision', 'recall', 'f1_score'], cmap='Reds')
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Consensus anomalies (detected by multiple methods)
                    if len(anomaly_results) > 1:
                        consensus_anomalies = np.zeros(len(X), dtype=bool)
                        for anomalies in anomaly_results.values():
                            consensus_anomalies |= anomalies
                        
                        # Count how many methods agree
                        agreement_count = np.zeros(len(X))
                        for anomalies in anomaly_results.values():
                            agreement_count += anomalies.astype(int)
                        
                        high_consensus = agreement_count >= len(anomaly_results) // 2 + 1
                        
                        st.subheader("Consensus Analysis")
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric("Total Unique Anomalies", consensus_anomalies.sum())
                        with col_b:
                            st.metric("High Consensus Anomalies", high_consensus.sum())
                        with col_c:
                            if high_consensus.sum() > 0:
                                consensus_fraud_rate = y[high_consensus].mean()
                                st.metric("Consensus Fraud Rate", f"{consensus_fraud_rate:.2%}")
    
    with col2:
        st.markdown("### Suite Guidelines")
        st.info("""
        **Comprehensive Analysis:**
        
        **Multiple Methods:**
        - Compare different approaches
        - Find consensus anomalies
        - Validate detection quality
        
        **Method Strengths:**
        - Isolation Forest: Fast, scalable
        - One-Class SVM: Complex boundaries
        - LOF: Local density analysis
        - Elliptic Envelope: Gaussian data
        
        **Best Practices:**
        - Use multiple methods
        - Look for consensus
        - Validate with domain knowledge
        - Consider ensemble approaches
        """)

def _render_association_analysis():
    """Render association analysis."""
    st.subheader("Association Analysis")
    
    st.markdown("""
    ### Feature Association & Pattern Mining
    
    Discover associations between features and identify patterns
    that distinguish fraudulent from legitimate transactions.
    """)
    
    st.info("""
    **Association Analysis Components:**
    
    📊 **Feature Correlation Analysis:** Identify highly correlated features
    🔗 **Pattern Mining:** Find frequent patterns in fraud vs legitimate transactions  
    📈 **Statistical Dependencies:** Measure feature dependencies and interactions
    🎯 **Fraud Signature Discovery:** Identify unique fraud-specific patterns
    
    **Note:** Advanced association rule mining requires specialized libraries.
    This section provides correlation and statistical dependency analysis.
    """)
    
    if st.button("Run Association Analysis", key="association_btn"):
        X, y = load_data(use_synthetic=True)
        
        # Create DataFrame for easier analysis
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['is_fraud'] = y
        
        # Correlation analysis
        st.subheader("Feature Correlation Analysis")
        
        corr_matrix = df.corr()
        
        # Plot correlation heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Full correlation matrix
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, ax=ax1)
        ax1.set_title('Full Feature Correlation Matrix')
        
        # Correlation with fraud
        fraud_corr = corr_matrix['is_fraud'].drop('is_fraud').sort_values(key=abs, ascending=False)
        
        # Top correlations with fraud
        top_n = min(20, len(fraud_corr))
        top_fraud_corr = fraud_corr.head(top_n)
        
        colors = ['red' if x < 0 else 'green' for x in top_fraud_corr.values]
        bars = ax2.barh(range(len(top_fraud_corr)), top_fraud_corr.values, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(top_fraud_corr)))
        ax2.set_yticklabels(top_fraud_corr.index)
        ax2.set_xlabel('Correlation with Fraud')
        ax2.set_title('Top Feature-Fraud Correlations')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Statistical summary
        st.subheader("Correlation Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            highest_pos_corr = fraud_corr.max()
            st.metric("Highest Positive Correlation", f"{highest_pos_corr:.4f}")
        
        with col2:
            highest_neg_corr = fraud_corr.min()
            st.metric("Highest Negative Correlation", f"{highest_neg_corr:.4f}")
        
        with col3:
            strong_correlations = (abs(fraud_corr) > 0.3).sum()
            st.metric("Strong Correlations (>0.3)", strong_correlations)
        
        with col4:
            avg_abs_corr = abs(fraud_corr).mean()
            st.metric("Average Absolute Correlation", f"{avg_abs_corr:.4f}")
        
        # Feature importance based on correlation
        st.subheader("Most Discriminative Features")
        
        top_features = abs(fraud_corr).sort_values(ascending=False).head(10)
        
        for i, (feature, corr_value) in enumerate(top_features.items(), 1):
            direction = "positively" if fraud_corr[feature] > 0 else "negatively"
            st.write(f"{i}. **{feature}**: {direction} correlated ({fraud_corr[feature]:.4f})")

def _render_unsupervised_model_comparison():
    """Render comprehensive unsupervised model comparison."""
    st.subheader("Unsupervised Model Comparison")
    
    st.markdown("""
    ### Comprehensive Model Performance Analysis
    
    Compare different unsupervised learning models across multiple metrics
    to identify the best approach for fraud detection.
    """)
    
    if st.button("Run Comprehensive Model Comparison", key="model_comparison_btn"):
        X, y = load_data(use_synthetic=True)
        
        # Preprocess data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Define models to compare
        models = {
            "K-Means (3)": KMeans(n_clusters=3, random_state=42),
            "K-Means (5)": KMeans(n_clusters=5, random_state=42),
            "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
            "Agglomerative": AgglomerativeClustering(n_clusters=3),
            "Gaussian Mixture": GaussianMixture(n_components=3, random_state=42),
            "Isolation Forest": IsolationForest(contamination=0.1, random_state=42),
            "One-Class SVM": OneClassSVM(nu=0.1)
        }
        
        results = {}
        
        progress_bar = st.progress(0)
        
        for idx, (model_name, model) in enumerate(models.items()):
            try:
                if "Isolation" in model_name or "SVM" in model_name:
                    # Anomaly detection models
                    if "Isolation" in model_name:
                        predictions = model.fit(X_scaled).predict(X_scaled)
                    else:
                        predictions = model.fit(X_scaled).predict(X_scaled)
                    
                    anomalies = predictions == -1
                    n_anomalies = anomalies.sum()
                    
                    if n_anomalies > 0:
                        fraud_detection_rate = y[anomalies].mean()
                        precision = fraud_detection_rate
                        recall = y[anomalies].sum() / y.sum() if y.sum() > 0 else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    else:
                        fraud_detection_rate = precision = recall = f1 = 0
                    
                    results[model_name] = {
                        'type': 'Anomaly Detection',
                        'n_clusters': 2,  # Binary: normal/anomaly
                        'silhouette': np.nan,
                        'ari': adjusted_rand_score(y, anomalies.astype(int)),
                        'nmi': normalized_mutual_info_score(y, anomalies.astype(int)),
                        'fraud_detection_rate': fraud_detection_rate,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'anomaly_rate': n_anomalies / len(X)
                    }
                    
                else:
                    # Clustering models
                    labels = model.fit_predict(X_scaled)
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    
                    # Clustering quality metrics
                    if n_clusters > 1:
                        silhouette = silhouette_score(X_scaled, labels)
                    else:
                        silhouette = np.nan
                    
                    ari = adjusted_rand_score(y, labels)
                    nmi = normalized_mutual_info_score(y, labels)
                    
                    # Fraud detection effectiveness
                    fraud_rates = []
                    for cluster_id in set(labels):
                        if cluster_id != -1:
                            mask = labels == cluster_id
                            if mask.sum() > 0:
                                fraud_rate = y[mask].mean()
                                fraud_rates.append(fraud_rate)
                    
                    max_fraud_rate = max(fraud_rates) if fraud_rates else 0
                    
                    results[model_name] = {
                        'type': 'Clustering',
                        'n_clusters': n_clusters,
                        'silhouette': silhouette,
                        'ari': ari,
                        'nmi': nmi,
                        'fraud_detection_rate': max_fraud_rate,
                        'precision': np.nan,
                        'recall': np.nan,
                        'f1_score': np.nan,
                        'anomaly_rate': (labels == -1).sum() / len(labels) if -1 in labels else 0
                    }
                
            except Exception as e:
                st.warning(f"Error with {model_name}: {str(e)}")
                results[model_name] = {
                    'type': 'Error',
                    'n_clusters': 0,
                    'silhouette': np.nan,
                    'ari': 0,
                    'nmi': 0,
                    'fraud_detection_rate': 0,
                    'precision': np.nan,
                    'recall': np.nan,
                    'f1_score': np.nan,
                    'anomaly_rate': 0
                }
            
            progress_bar.progress((idx + 1) / len(models))
        
        # Create comprehensive results DataFrame
        results_df = pd.DataFrame(results).T
        results_df = results_df.round(4)
        
        # Display results
        st.subheader("Comprehensive Model Comparison Results")
        
        # Split by model type
        clustering_results = results_df[results_df['type'] == 'Clustering']
        anomaly_results = results_df[results_df['type'] == 'Anomaly Detection']
        
        if not clustering_results.empty:
            st.markdown("**Clustering Models:**")
            clustering_display = clustering_results.drop('type', axis=1)
            st.dataframe(clustering_display.style.background_gradient(subset=['silhouette', 'ari', 'nmi', 'fraud_detection_rate'], cmap='Greens'), use_container_width=True)
        
        if not anomaly_results.empty:
            st.markdown("**Anomaly Detection Models:**")
            anomaly_display = anomaly_results.drop('type', axis=1)
            st.dataframe(anomaly_display.style.background_gradient(subset=['precision', 'recall', 'f1_score', 'fraud_detection_rate'], cmap='Reds'), use_container_width=True)
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Fraud detection rates
        all_models = list(results.keys())
        fraud_rates = [results[model]['fraud_detection_rate'] for model in all_models]
        
        bars1 = axes[0,0].bar(range(len(all_models)), fraud_rates, alpha=0.7)
        axes[0,0].set_title('Fraud Detection Rate by Model')
        axes[0,0].set_ylabel('Max Fraud Rate in Cluster/Anomalies')
        axes[0,0].set_xticks(range(len(all_models)))
        axes[0,0].set_xticklabels(all_models, rotation=45, ha='right')
        
        # Plot 2: ARI scores
        ari_scores = [results[model]['ari'] for model in all_models]
        bars2 = axes[0,1].bar(range(len(all_models)), ari_scores, alpha=0.7, color='orange')
        axes[0,1].set_title('Adjusted Rand Index')
        axes[0,1].set_ylabel('ARI Score')
        axes[0,1].set_xticks(range(len(all_models)))
        axes[0,1].set_xticklabels(all_models, rotation=45, ha='right')
        
        # Plot 3: Silhouette scores (clustering only)
        clustering_models = [m for m in all_models if results[m]['type'] == 'Clustering']
        if clustering_models:
            sil_scores = [results[model]['silhouette'] for model in clustering_models]
            sil_scores = [s for s in sil_scores if not np.isnan(s)]
            if sil_scores:
                bars3 = axes[1,0].bar(range(len(clustering_models)), 
                                    [results[model]['silhouette'] for model in clustering_models], 
                                    alpha=0.7, color='green')
                axes[1,0].set_title('Silhouette Scores (Clustering Models)')
                axes[1,0].set_ylabel('Silhouette Score')
                axes[1,0].set_xticks(range(len(clustering_models)))
                axes[1,0].set_xticklabels(clustering_models, rotation=45, ha='right')
        
        # Plot 4: F1 scores (anomaly detection only)
        anomaly_models = [m for m in all_models if results[m]['type'] == 'Anomaly Detection']
        if anomaly_models:
            f1_scores = [results[model]['f1_score'] for model in anomaly_models]
            f1_scores = [s for s in f1_scores if not np.isnan(s)]
            if f1_scores:
                bars4 = axes[1,1].bar(range(len(anomaly_models)), 
                                    [results[model]['f1_score'] for model in anomaly_models], 
                                    alpha=0.7, color='red')
                axes[1,1].set_title('F1 Scores (Anomaly Detection Models)')
                axes[1,1].set_ylabel('F1 Score')
                axes[1,1].set_xticks(range(len(anomaly_models)))
                axes[1,1].set_xticklabels(anomaly_models, rotation=45, ha='right')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Recommendations
        st.subheader("Model Recommendations")
        
        # Best clustering model
        if not clustering_results.empty:
            clustering_scores = {}
            for model in clustering_results.index:
                score = (clustering_results.loc[model, 'silhouette'] * 0.3 + 
                        clustering_results.loc[model, 'ari'] * 0.3 + 
                        clustering_results.loc[model, 'fraud_detection_rate'] * 0.4)
                if not np.isnan(score):
                    clustering_scores[model] = score
            
            if clustering_scores:
                best_clustering = max(clustering_scores.keys(), key=lambda k: clustering_scores[k])
                st.success(f"🏆 **Best Clustering Model:** {best_clustering} (score: {clustering_scores[best_clustering]:.3f})")
        
        # Best anomaly detection model
        if not anomaly_results.empty:
            anomaly_scores = {}
            for model in anomaly_results.index:
                f1 = anomaly_results.loc[model, 'f1_score']
                if not np.isnan(f1):
                    anomaly_scores[model] = f1
            
            if anomaly_scores:
                best_anomaly = max(anomaly_scores.keys(), key=lambda k: anomaly_scores[k])
                st.success(f"🎯 **Best Anomaly Detection Model:** {best_anomaly} (F1: {anomaly_scores[best_anomaly]:.3f})")

def _render_supervised_vs_unsupervised():
    """Render supervised vs unsupervised comparison."""
    st.subheader("🤖 Supervised vs Unsupervised Comparison")
    
    st.markdown("""
    ### Performance Comparison Between Learning Paradigms
    
    Compare the effectiveness of supervised and unsupervised approaches
    for fraud detection to understand their relative strengths.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Model Selection")
        
        # Supervised models selection
        supervised_models = get_supervised_models()
        selected_supervised = st.multiselect(
            "Select supervised models",
            options=list(supervised_models.keys()),
            default=list(supervised_models.keys())[:2],
            key="svsu_supervised"
        )
        
        # Unsupervised models selection
        unsupervised_models = get_unsupervised_models()
        selected_unsupervised = st.multiselect(
            "Select unsupervised models",
            options=list(unsupervised_models.keys()),
            default=list(unsupervised_models.keys())[:2],
            key="svsu_unsupervised"
        )
        
        if st.button("Compare Supervised vs Unsupervised", key="svsu_compare_btn"):
            if selected_supervised or selected_unsupervised:
                X, y = load_data(use_synthetic=True)
                
                # Split data for supervised evaluation
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                # Preprocess data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                results = {}
                
                # Evaluate supervised models
                for model_name in selected_supervised:
                    model = supervised_models[model_name]
                    model.fit(X_train, y_train)
                    
                    # Get predictions and probabilities
                    y_pred = model.predict(X_test)
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_test)[:, 1]
                    else:
                        y_proba = model.decision_function(X_test)
                    
                    # Calculate metrics
                    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    auc = roc_auc_score(y_test, y_proba)
                    
                    results[f"{model_name} (Supervised)"] = {
                        'type': 'Supervised',
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'auc': auc,
                        'accuracy': (y_pred == y_test).mean(),
                        'requires_labels': True
                    }
                
                # Evaluate unsupervised models
                for model_name in selected_unsupervised:
                    model = unsupervised_models[model_name]
                    
                    # Fit on training data, predict on test data
                    model.fit(X_train_scaled)
                    
                    if hasattr(model, 'predict'):
                        predictions = model.predict(X_test_scaled)
                        # Convert clustering labels to anomaly scores
                        if model_name in ['K-Means', 'DBSCAN', 'Agglomerative Clustering', 'Gaussian Mixture']:
                            # For clustering: find cluster with highest fraud rate
                            unique_labels = set(predictions)
                            cluster_fraud_rates = {}
                            for label in unique_labels:
                                if label != -1:  # Skip noise points
                                    mask = predictions == label
                                    if mask.sum() > 0:
                                        fraud_rate = y_test[mask].mean()
                                        cluster_fraud_rates[label] = fraud_rate
                            
                            if cluster_fraud_rates:
                                # Predict fraud for points in highest fraud rate cluster
                                best_cluster = max(cluster_fraud_rates.keys(), key=lambda k: cluster_fraud_rates[k])
                                y_pred = (predictions == best_cluster).astype(int)
                            else:
                                y_pred = np.zeros(len(y_test))
                        else:
                            # For anomaly detection: -1 indicates anomaly
                            y_pred = (predictions == -1).astype(int)
                    else:
                        # Fallback for models without predict method
                        y_pred = np.zeros(len(y_test))
                    
                    # Calculate metrics
                    if y_pred.sum() > 0:  # If any positive predictions
                        precision = precision_score(y_test, y_pred, zero_division=0)
                        recall = recall_score(y_test, y_pred, zero_division=0)
                        f1 = f1_score(y_test, y_pred, zero_division=0)
                    else:
                        precision = recall = f1 = 0
                    
                    results[f"{model_name} (Unsupervised)"] = {
                        'type': 'Unsupervised',
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'auc': np.nan,  # Not applicable for unsupervised
                        'accuracy': (y_pred == y_test).mean(),
                        'requires_labels': False
                    }
                
                # Display results
                st.subheader("Comparison Results")
                
                # Create comparison DataFrame
                comparison_df = pd.DataFrame(results).T
                comparison_df = comparison_df.round(4)
                
                # Split by type for better visualization
                supervised_results = comparison_df[comparison_df['type'] == 'Supervised']
                unsupervised_results = comparison_df[comparison_df['type'] == 'Unsupervised']
                
                if not supervised_results.empty:
                    st.markdown("**Supervised Models:**")
                    sup_display = supervised_results.drop(['type', 'requires_labels'], axis=1)
                    st.dataframe(sup_display.style.background_gradient(subset=['f1_score', 'auc'], cmap='Blues'), use_container_width=True)
                
                if not unsupervised_results.empty:
                    st.markdown("**Unsupervised Models:**")
                    unsup_display = unsupervised_results.drop(['type', 'requires_labels', 'auc'], axis=1)
                    st.dataframe(unsup_display.style.background_gradient(subset=['f1_score'], cmap='Greens'), use_container_width=True)
                
                # Visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                models = list(results.keys())
                f1_scores = [results[model]['f1_score'] for model in models]
                precisions = [results[model]['precision'] for model in models]
                recalls = [results[model]['recall'] for model in models]
                
                # Color by type
                colors = ['blue' if 'Supervised' in model else 'green' for model in models]
                
                # Plot 1: F1 Scores
                bars1 = ax1.bar(range(len(models)), f1_scores, color=colors, alpha=0.7)
                ax1.set_title('F1 Score Comparison')
                ax1.set_ylabel('F1 Score')
                ax1.set_xticks(range(len(models)))
                ax1.set_xticklabels(models, rotation=45, ha='right')
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Precision vs Recall
                supervised_mask = [i for i, model in enumerate(models) if 'Supervised' in model]
                unsupervised_mask = [i for i, model in enumerate(models) if 'Unsupervised' in model]
                
                if supervised_mask:
                    sup_prec = [precisions[i] for i in supervised_mask]
                    sup_rec = [recalls[i] for i in supervised_mask]
                    ax2.scatter(sup_rec, sup_prec, c='blue', label='Supervised', s=100, alpha=0.7)
                
                if unsupervised_mask:
                    unsup_prec = [precisions[i] for i in unsupervised_mask]
                    unsup_rec = [recalls[i] for i in unsupervised_mask]
                    ax2.scatter(unsup_rec, unsup_prec, c='green', label='Unsupervised', s=100, alpha=0.7)
                
                ax2.set_xlabel('Recall')
                ax2.set_ylabel('Precision')
                ax2.set_title('Precision vs Recall')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Summary insights
                st.subheader("Key Insights")
                
                if not supervised_results.empty and not unsupervised_results.empty:
                    best_supervised_f1 = supervised_results['f1_score'].max()
                    best_unsupervised_f1 = unsupervised_results['f1_score'].max()
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Best Supervised F1", f"{best_supervised_f1:.3f}")
                    with col_b:
                        st.metric("Best Unsupervised F1", f"{best_unsupervised_f1:.3f}")
                    with col_c:
                        improvement = (best_supervised_f1 - best_unsupervised_f1) / best_unsupervised_f1 * 100 if best_unsupervised_f1 > 0 else 0
                        st.metric("Supervised Advantage", f"{improvement:.1f}%")
    
    with col2:
        st.markdown("### Comparison Framework")
        st.info("""
        **Supervised Advantages:**
        - Uses labeled data for training
        - Generally higher accuracy
        - Direct optimization for task
        - Well-established metrics
        
        **Unsupervised Advantages:**
        - No labeled data required
        - Discovers hidden patterns
        - Detects novel fraud types
        - Less prone to label bias
        
        **Trade-offs:**
        - Supervised: Requires expensive labels
        - Unsupervised: Lower precision typically
        - Hybrid approaches often work best
        """)

def _render_pattern_discovery():
    """Render pattern discovery analysis."""
    st.subheader("📊 Pattern Discovery")
    
    st.markdown("""
    ### Automated Pattern Discovery in Fraud Data
    
    Use unsupervised techniques to automatically discover patterns
    and structures in the data that may indicate fraudulent behavior.
    """)
    
    # Pattern discovery subtabs
    pattern_tabs = st.tabs([
        "Statistical Patterns",
        "Clustering Patterns", 
        "Anomaly Patterns",
        "Temporal Patterns"
    ])
    
    with pattern_tabs[0]:
        st.subheader("Statistical Pattern Analysis")
        
        if st.button("Discover Statistical Patterns", key="stat_patterns_btn"):
            X, y = load_data(use_synthetic=True)
            
            # Feature statistics for fraud vs legitimate
            fraud_mask = y == 1
            legit_mask = y == 0
            
            X_fraud = X[fraud_mask]
            X_legit = X[legit_mask]
            
            # Calculate statistical differences
            patterns = []
            for i in range(X.shape[1]):
                fraud_mean = X_fraud[:, i].mean()
                legit_mean = X_legit[:, i].mean()
                fraud_std = X_fraud[:, i].std()
                legit_std = X_legit[:, i].std()
                
                # Statistical significance test
                from scipy.stats import ttest_ind
                t_stat, p_value = ttest_ind(X_fraud[:, i], X_legit[:, i])
                
                patterns.append({
                    'Feature': f'Feature_{i}',
                    'Fraud_Mean': fraud_mean,
                    'Legit_Mean': legit_mean,
                    'Difference': fraud_mean - legit_mean,
                    'Fraud_Std': fraud_std,
                    'Legit_Std': legit_std,
                    'T_Statistic': t_stat,
                    'P_Value': p_value,
                    'Significant': p_value < 0.05
                })
            
            patterns_df = pd.DataFrame(patterns)
            significant_patterns = patterns_df[patterns_df['Significant']].sort_values('P_Value')
            
            st.subheader("Significant Statistical Patterns")
            st.dataframe(significant_patterns.head(10).round(4), use_container_width=True)
            
            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Mean differences
            top_features = significant_patterns.head(10)
            ax1.barh(range(len(top_features)), top_features['Difference'].values, alpha=0.7)
            ax1.set_yticks(range(len(top_features)))
            ax1.set_yticklabels(top_features['Feature'].values)
            ax1.set_xlabel('Mean Difference (Fraud - Legitimate)')
            ax1.set_title('Top Feature Mean Differences')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: P-values
            ax2.barh(range(len(top_features)), -np.log10(top_features['P_Value'].values), alpha=0.7, color='red')
            ax2.set_yticks(range(len(top_features)))
            ax2.set_yticklabels(top_features['Feature'].values)
            ax2.set_xlabel('-log10(P-Value)')
            ax2.set_title('Statistical Significance')
            ax2.axvline(x=-np.log10(0.05), color='black', linestyle='--', alpha=0.5, label='p=0.05')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    with pattern_tabs[1]:
        st.subheader("Clustering-Based Patterns")
        
        if st.button("Discover Clustering Patterns", key="cluster_patterns_btn"):
            X, y = load_data(use_synthetic=True)
            
            # Preprocess data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=5, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Analyze patterns in each cluster
            cluster_analysis = []
            for cluster_id in range(5):
                mask = cluster_labels == cluster_id
                cluster_size = mask.sum()
                fraud_rate = y[mask].mean()
                
                # Feature characteristics
                cluster_features = X[mask].mean(axis=0)
                
                cluster_analysis.append({
                    'Cluster': cluster_id,
                    'Size': cluster_size,
                    'Fraud_Rate': fraud_rate,
                    'Size_Percent': cluster_size / len(X) * 100,
                    'Risk_Level': 'High' if fraud_rate > 0.2 else 'Medium' if fraud_rate > 0.1 else 'Low'
                })
            
            cluster_df = pd.DataFrame(cluster_analysis)
            cluster_df = cluster_df.sort_values('Fraud_Rate', ascending=False)
            
            st.subheader("Cluster Pattern Analysis")
            st.dataframe(cluster_df.round(4), use_container_width=True)
            
            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Cluster sizes and fraud rates
            bars = ax1.bar(cluster_df['Cluster'], cluster_df['Size'], alpha=0.7, color='lightblue')
            ax1_twin = ax1.twinx()
            line = ax1_twin.plot(cluster_df['Cluster'], cluster_df['Fraud_Rate'], 'ro-', linewidth=2, markersize=8)
            
            ax1.set_xlabel('Cluster ID')
            ax1.set_ylabel('Cluster Size', color='blue')
            ax1_twin.set_ylabel('Fraud Rate', color='red')
            ax1.set_title('Cluster Size vs Fraud Rate')
            
            # Plot 2: Risk distribution
            risk_counts = cluster_df['Risk_Level'].value_counts()
            ax2.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Risk Level Distribution')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    with pattern_tabs[2]:
        st.subheader("Anomaly Patterns")
        
        if st.button("Discover Anomaly Patterns", key="anomaly_patterns_btn"):
            X, y = load_data(use_synthetic=True)
            
            # Preprocess data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_scores = iso_forest.fit(X_scaled).decision_function(X_scaled)
            anomalies = iso_forest.predict(X_scaled) == -1
            
            # Analyze anomaly patterns
            anomaly_analysis = {
                'Total_Anomalies': anomalies.sum(),
                'Anomaly_Rate': anomalies.mean(),
                'Fraud_in_Anomalies': y[anomalies].sum() if anomalies.sum() > 0 else 0,
                'Fraud_Rate_in_Anomalies': y[anomalies].mean() if anomalies.sum() > 0 else 0,
                'Overall_Fraud_Rate': y.mean(),
                'Anomaly_Effectiveness': (y[anomalies].mean() / y.mean()) if y.mean() > 0 and anomalies.sum() > 0 else 0
            }
            
            st.subheader("Anomaly Pattern Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Anomalies", anomaly_analysis['Total_Anomalies'])
            with col2:
                st.metric("Anomaly Rate", f"{anomaly_analysis['Anomaly_Rate']:.2%}")
            with col3:
                st.metric("Fraud in Anomalies", f"{anomaly_analysis['Fraud_Rate_in_Anomalies']:.2%}")
            with col4:
                st.metric("Detection Effectiveness", f"{anomaly_analysis['Anomaly_Effectiveness']:.2f}x")
            
            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Anomaly score distribution
            ax1.hist(anomaly_scores[y==0], bins=50, alpha=0.7, label='Legitimate', color='blue')
            ax1.hist(anomaly_scores[y==1], bins=50, alpha=0.7, label='Fraud', color='red')
            ax1.set_xlabel('Anomaly Score')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Anomaly Score Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: 2D visualization of anomalies
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            
            normal_mask = ~anomalies
            ax2.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], c='blue', alpha=0.6, label='Normal', s=20)
            ax2.scatter(X_pca[anomalies, 0], X_pca[anomalies, 1], c='red', alpha=0.8, label='Anomaly', s=30)
            ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            ax2.set_title('Anomaly Detection Visualization')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    with pattern_tabs[3]:
        st.subheader("Temporal Patterns")
        st.info("""
        **Temporal Pattern Analysis:**
        
        In a real-world scenario, this would analyze:
        - Transaction timing patterns
        - Seasonal fraud trends
        - Weekly/daily patterns
        - Burst detection in fraudulent activity
        
        **Note:** This requires timestamp data which is not available in the current synthetic dataset.
        """)

def _render_cluster_based_insights():
    """Render cluster-based insights analysis."""
    st.subheader("📈 Cluster-Based Insights")
    
    st.markdown("""
    ### Deep Dive into Cluster Characteristics
    
    Analyze cluster properties to understand what makes each cluster unique
    and how they relate to fraud patterns.
    """)
    
    # Cluster insights subtabs
    cluster_tabs = st.tabs([
        "Cluster Profiling",
        "Feature Analysis",
        "Fraud Characterization",
        "Cluster Evolution"
    ])
    
    with cluster_tabs[0]:
        st.subheader("Cluster Profiling")
        
        n_clusters = st.slider("Number of clusters", 3, 8, 5, key="profile_clusters")
        
        if st.button("Generate Cluster Profiles", key="cluster_profile_btn"):
            X, y = load_data(use_synthetic=True)
            
            # Preprocess data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Create comprehensive cluster profiles
            profiles = []
            for cluster_id in range(n_clusters):
                mask = cluster_labels == cluster_id
                cluster_data = X[mask]
                cluster_labels_y = y[mask]
                
                profile = {
                    'Cluster_ID': cluster_id,
                    'Size': mask.sum(),
                    'Size_Percent': mask.sum() / len(X) * 100,
                    'Fraud_Count': cluster_labels_y.sum(),
                    'Fraud_Rate': cluster_labels_y.mean(),
                    'Avg_Feature_0': cluster_data[:, 0].mean(),
                    'Avg_Feature_1': cluster_data[:, 1].mean(),
                    'Avg_Feature_2': cluster_data[:, 2].mean() if cluster_data.shape[1] > 2 else 0,
                    'Feature_Variance': cluster_data.var(axis=0).mean(),
                    'Intra_Cluster_Distance': np.mean([np.linalg.norm(point - kmeans.cluster_centers_[cluster_id]) 
                                                     for point in X_scaled[mask]])
                }
                profiles.append(profile)
            
            profile_df = pd.DataFrame(profiles)
            profile_df = profile_df.sort_values('Fraud_Rate', ascending=False)
            
            st.subheader("Cluster Profiles")
            st.dataframe(profile_df.round(4), use_container_width=True)
            
            # Visualizations
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Cluster sizes
            ax1.bar(profile_df['Cluster_ID'], profile_df['Size'], alpha=0.7, color='lightblue')
            ax1.set_xlabel('Cluster ID')
            ax1.set_ylabel('Cluster Size')
            ax1.set_title('Cluster Size Distribution')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Fraud rates
            bars = ax2.bar(profile_df['Cluster_ID'], profile_df['Fraud_Rate'], alpha=0.7, color='red')
            ax2.set_xlabel('Cluster ID')
            ax2.set_ylabel('Fraud Rate')
            ax2.set_title('Fraud Rate by Cluster')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, rate in zip(bars, profile_df['Fraud_Rate']):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{rate:.3f}', ha='center', va='bottom')
            
            # Plot 3: Feature variance
            ax3.bar(profile_df['Cluster_ID'], profile_df['Feature_Variance'], alpha=0.7, color='green')
            ax3.set_xlabel('Cluster ID')
            ax3.set_ylabel('Average Feature Variance')
            ax3.set_title('Cluster Internal Variance')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Intra-cluster distances
            ax4.bar(profile_df['Cluster_ID'], profile_df['Intra_Cluster_Distance'], alpha=0.7, color='orange')
            ax4.set_xlabel('Cluster ID')
            ax4.set_ylabel('Average Distance to Center')
            ax4.set_title('Cluster Compactness')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Insights
            st.subheader("Key Insights")
            
            highest_fraud_cluster = profile_df.loc[profile_df['Fraud_Rate'].idxmax()]
            largest_cluster = profile_df.loc[profile_df['Size'].idxmax()]
            most_compact_cluster = profile_df.loc[profile_df['Intra_Cluster_Distance'].idxmin()]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Highest Risk Cluster", 
                         f"Cluster {highest_fraud_cluster['Cluster_ID']}", 
                         f"{highest_fraud_cluster['Fraud_Rate']:.2%} fraud rate")
            
            with col2:
                st.metric("Largest Cluster", 
                         f"Cluster {largest_cluster['Cluster_ID']}", 
                         f"{largest_cluster['Size_Percent']:.1f}% of data")
            
            with col3:
                st.metric("Most Compact Cluster", 
                         f"Cluster {most_compact_cluster['Cluster_ID']}", 
                         f"{most_compact_cluster['Intra_Cluster_Distance']:.3f} avg distance")
    
    with cluster_tabs[1]:
        st.subheader("Feature Analysis per Cluster")
        st.info("Detailed feature analysis for each cluster would be implemented here, showing which features are most characteristic of each cluster.")
    
    with cluster_tabs[2]:
        st.subheader("Fraud Characterization")
        st.info("Analysis of fraud patterns within and across clusters, including fraud signature identification.")
    
    with cluster_tabs[3]:
        st.subheader("Cluster Evolution")
        st.info("Analysis of how clusters change over time or with different parameters, including stability assessment.")
