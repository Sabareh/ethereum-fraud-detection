import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (confusion_matrix, roc_curve, auc, precision_recall_curve, 
                           average_precision_score, f1_score, precision_score, recall_score,
                           balanced_accuracy_score, accuracy_score, matthews_corrcoef, 
                           cohen_kappa_score, silhouette_score, adjusted_rand_score)
from sklearn.preprocessing import StandardScaler

from model_utils import get_supervised_models, get_unsupervised_models, load_data

def render_performance_analysis_tab():
    """Render comprehensive performance analysis tab."""
    st.header("🎯 Performance Analysis")
    
    st.markdown("""
    ### Comprehensive Model Performance Evaluation
    
    This section provides detailed performance analysis tools to evaluate model effectiveness
    across multiple dimensions and identify potential issues or improvements.
    """)
    
    # Performance analysis subtabs
    perf_tabs = st.tabs([
        "Supervised Performance",
        "Unsupervised Performance", 
        "Cross-Model Comparison",
        "Threshold Optimization",
        "Performance Metrics Deep Dive"
    ])
    
    with perf_tabs[0]:
        _render_supervised_performance()
    
    with perf_tabs[1]:
        _render_unsupervised_performance()
    
    with perf_tabs[2]:
        _render_cross_model_comparison()
    
    with perf_tabs[3]:
        _render_threshold_optimization()
    
    with perf_tabs[4]:
        _render_performance_metrics_deep_dive()

def _render_supervised_performance():
    """Render supervised model performance analysis."""
    st.subheader("Supervised Model Performance")
    
    # Existing supervised performance subtabs
    sup_tabs = st.tabs([
        "Confusion Matrix Analysis",
        "ROC & Precision-Recall",
        "Cross-Validation Results"
    ])
    
    with sup_tabs[0]:
        _render_confusion_matrix_analysis()
    
    with sup_tabs[1]:
        _render_roc_pr_analysis()
    
    with sup_tabs[2]:
        _render_cross_validation_analysis()

def _render_unsupervised_performance():
    """Render unsupervised model performance analysis."""
    st.subheader("Unsupervised Model Performance")
    
    st.markdown("""
    ### Clustering Performance Evaluation
    
    Analyze unsupervised models using clustering quality metrics and fraud detection effectiveness.
    """)
    
    # Unsupervised performance subtabs
    unsup_tabs = st.tabs([
        "Clustering Quality",
        "Fraud Detection Analysis",
        "Cluster Visualization",
        "Stability Analysis"
    ])
    
    with unsup_tabs[0]:
        _render_clustering_quality_analysis()
    
    with unsup_tabs[1]:
        _render_fraud_detection_analysis()
    
    with unsup_tabs[2]:
        _render_cluster_visualization_analysis()
    
    with unsup_tabs[3]:
        _render_clustering_stability_analysis()

def _render_clustering_quality_analysis():
    """Render clustering quality analysis."""
    st.subheader("Clustering Quality Metrics")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        unsupervised_models = get_unsupervised_models()
        selected_models = st.multiselect(
            "Select unsupervised models for analysis",
            options=list(unsupervised_models.keys()),
            default=list(unsupervised_models.keys())[:3],
            key="clustering_quality_models"
        )
        
        use_pca = st.checkbox("Apply PCA preprocessing", value=True)
        n_components = st.slider("PCA components", 2, 10, 3) if use_pca else None
        
        if st.button("Analyze Clustering Quality", key="clustering_quality_btn"):
            if selected_models:
                X, y = load_data(use_synthetic=True)
                
                # Preprocess data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                if use_pca:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=n_components, random_state=42)
                    X_processed = pca.fit_transform(X_scaled)
                else:
                    X_processed = X_scaled
                
                results = []
                
                for model_name in selected_models:
                    try:
                        model = unsupervised_models[model_name]
                        labels = model.fit_predict(X_processed)
                        
                        # Calculate clustering metrics
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        
                        if n_clusters > 1:
                            silhouette = silhouette_score(X_processed, labels)
                            calinski_harabasz = None
                            davies_bouldin = None
                            
                            try:
                                from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
                                calinski_harabasz = calinski_harabasz_score(X_processed, labels)
                                davies_bouldin = davies_bouldin_score(X_processed, labels)
                            except:
                                pass
                        else:
                            silhouette = np.nan
                            calinski_harabasz = np.nan
                            davies_bouldin = np.nan
                        
                        # Noise ratio (for DBSCAN-like algorithms)
                        noise_ratio = np.sum(labels == -1) / len(labels) if -1 in labels else 0
                        
                        results.append({
                            'Model': model_name,
                            'N_Clusters': n_clusters,
                            'Silhouette_Score': silhouette,
                            'Calinski_Harabasz': calinski_harabasz,
                            'Davies_Bouldin': davies_bouldin,
                            'Noise_Ratio': noise_ratio
                        })
                        
                    except Exception as e:
                        st.warning(f"Error analyzing {model_name}: {str(e)}")
                
                # Display results
                if results:
                    results_df = pd.DataFrame(results)
                    
                    # Format the dataframe for display
                    display_df = results_df.copy()
                    for col in ['Silhouette_Score', 'Calinski_Harabasz', 'Davies_Bouldin', 'Noise_Ratio']:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].apply(
                                lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A"
                            )
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Visualize metrics
                    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                    
                    # Silhouette scores
                    valid_silhouette = results_df.dropna(subset=['Silhouette_Score'])
                    if not valid_silhouette.empty:
                        axes[0,0].bar(valid_silhouette['Model'], valid_silhouette['Silhouette_Score'])
                        axes[0,0].set_title('Silhouette Scores')
                        axes[0,0].set_ylabel('Score')
                        axes[0,0].tick_params(axis='x', rotation=45)
                    
                    # Number of clusters
                    axes[0,1].bar(results_df['Model'], results_df['N_Clusters'])
                    axes[0,1].set_title('Number of Clusters')
                    axes[0,1].set_ylabel('Count')
                    axes[0,1].tick_params(axis='x', rotation=45)
                    
                    # Calinski-Harabasz scores
                    valid_ch = results_df.dropna(subset=['Calinski_Harabasz'])
                    if not valid_ch.empty:
                        axes[1,0].bar(valid_ch['Model'], valid_ch['Calinski_Harabasz'])
                        axes[1,0].set_title('Calinski-Harabasz Scores')
                        axes[1,0].set_ylabel('Score')
                        axes[1,0].tick_params(axis='x', rotation=45)
                    
                    # Noise ratio
                    axes[1,1].bar(results_df['Model'], results_df['Noise_Ratio'])
                    axes[1,1].set_title('Noise Ratio')
                    axes[1,1].set_ylabel('Ratio')
                    axes[1,1].tick_params(axis='x', rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
    
    with col2:
        st.markdown("### Clustering Metrics Guide")
        st.info("""
        **Silhouette Score (-1 to 1):**
        - Higher is better
        - >0.5: Good clusters
        - 0.2-0.5: Reasonable
        - <0.2: Poor clustering
        
        **Calinski-Harabasz:**
        - Higher is better
        - Ratio of between/within cluster variance
        
        **Davies-Bouldin:**
        - Lower is better
        - Average similarity between clusters
        
        **Noise Ratio:**
        - Lower usually better
        - Proportion of outliers detected
        """)

def _render_fraud_detection_analysis():
    """Render fraud detection effectiveness analysis for unsupervised models."""
    st.subheader("Fraud Detection Effectiveness")
    
    st.markdown("""
    ### How Well Do Clusters Identify Fraud?
    
    Analyze how effectively unsupervised models separate fraudulent from legitimate transactions.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        unsupervised_models = get_unsupervised_models()
        selected_models = st.multiselect(
            "Select models for fraud detection analysis",
            options=list(unsupervised_models.keys()),
            default=list(unsupervised_models.keys())[:3],
            key="fraud_detection_models"
        )
        
        if st.button("Analyze Fraud Detection", key="fraud_detection_btn"):
            if selected_models:
                X, y = load_data(use_synthetic=True)
                
                # Preprocess data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                results = []
                cluster_details = {}
                
                for model_name in selected_models:
                    try:
                        model = unsupervised_models[model_name]
                        labels = model.fit_predict(X_scaled)
                        
                        # Calculate fraud detection metrics
                        unique_labels = set(labels)
                        cluster_fraud_rates = {}
                        
                        for cluster_id in unique_labels:
                            if cluster_id != -1:  # Exclude noise
                                mask = labels == cluster_id
                                if mask.sum() > 0:
                                    fraud_rate = y[mask].mean()
                                    cluster_fraud_rates[cluster_id] = fraud_rate
                        
                        # Overall metrics
                        if cluster_fraud_rates:
                            max_fraud_rate = max(cluster_fraud_rates.values())
                            avg_fraud_rate = np.mean(list(cluster_fraud_rates.values()))
                        else:
                            max_fraud_rate = 0
                            avg_fraud_rate = 0
                        
                        # Adjusted Rand Index with true labels
                        ari_score = adjusted_rand_score(y, labels)
                        
                        # Separation effectiveness
                        fraud_separation = max_fraud_rate - y.mean()  # How much better than random
                        
                        results.append({
                            'Model': model_name,
                            'Max_Fraud_Rate': max_fraud_rate,
                            'Avg_Fraud_Rate': avg_fraud_rate,
                            'ARI_Score': ari_score,
                            'Fraud_Separation': fraud_separation,
                            'N_Clusters': len(cluster_fraud_rates)
                        })
                        
                        cluster_details[model_name] = cluster_fraud_rates
                        
                    except Exception as e:
                        st.warning(f"Error analyzing {model_name}: {str(e)}")
                
                # Display results
                if results:
                    results_df = pd.DataFrame(results)
                    
                    # Format for display
                    display_df = results_df.copy()
                    for col in ['Max_Fraud_Rate', 'Avg_Fraud_Rate', 'ARI_Score', 'Fraud_Separation']:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Visualizations
                    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                    
                    # Max fraud rate by model
                    axes[0,0].bar(results_df['Model'], results_df['Max_Fraud_Rate'])
                    axes[0,0].axhline(y=y.mean(), color='r', linestyle='--', label='Overall Fraud Rate')
                    axes[0,0].set_title('Maximum Fraud Rate by Cluster')
                    axes[0,0].set_ylabel('Fraud Rate')
                    axes[0,0].legend()
                    axes[0,0].tick_params(axis='x', rotation=45)
                    
                    # ARI scores
                    axes[0,1].bar(results_df['Model'], results_df['ARI_Score'])
                    axes[0,1].set_title('Adjusted Rand Index')
                    axes[0,1].set_ylabel('ARI Score')
                    axes[0,1].tick_params(axis='x', rotation=45)
                    
                    # Fraud separation effectiveness
                    axes[1,0].bar(results_df['Model'], results_df['Fraud_Separation'])
                    axes[1,0].set_title('Fraud Separation Effectiveness')
                    axes[1,0].set_ylabel('Improvement over Random')
                    axes[1,0].tick_params(axis='x', rotation=45)
                    
                    # Fraud rate distribution across clusters
                    if cluster_details:
                        all_rates = []
                        model_labels = []
                        for model_name, rates in cluster_details.items():
                            all_rates.extend(rates.values())
                            model_labels.extend([model_name] * len(rates))
                        
                        # Create box plot data
                        unique_models = list(cluster_details.keys())
                        box_data = [list(cluster_details[model].values()) for model in unique_models]
                        
                        axes[1,1].boxplot(box_data, labels=unique_models)
                        axes[1,1].set_title('Fraud Rate Distribution by Model')
                        axes[1,1].set_ylabel('Fraud Rate')
                        axes[1,1].tick_params(axis='x', rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Best performing model
                    best_model = results_df.loc[results_df['Fraud_Separation'].idxmax(), 'Model']
                    st.success(f"🏆 Best fraud detection: **{best_model}**")
    
    with col2:
        st.markdown("### Fraud Detection Metrics")
        st.info("""
        **Max Fraud Rate:**
        - Highest fraud concentration in any cluster
        - Higher indicates better separation
        
        **ARI Score:**
        - Agreement with true fraud labels
        - 1.0 = perfect agreement
        - 0.0 = random clustering
        
        **Fraud Separation:**
        - How much better than random detection
        - Positive values indicate improvement
        
        **Best Strategy:**
        - Focus on clusters with highest fraud rates
        - Use as anomaly detection signals
        """)

def _render_cluster_visualization_analysis():
    """Render cluster visualization analysis."""
    st.subheader("Cluster Visualization")
    
    st.markdown("""
    ### Visual Analysis of Clustering Results
    
    Visualize how different models cluster the data and identify fraud patterns.
    """)
    
    unsupervised_models = get_unsupervised_models()
    selected_model = st.selectbox(
        "Select model for visualization",
        options=list(unsupervised_models.keys()),
        key="viz_model"
    )
    
    if st.button("Generate Cluster Visualization", key="cluster_viz_btn"):
        X, y = load_data(use_synthetic=True)
        
        # Preprocess data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        # Fit the model
        model = unsupervised_models[selected_model]
        labels = model.fit_predict(X_scaled)
        
        # Create visualizations
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Clusters colored by cluster assignment
        scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.6)
        axes[0].set_title(f'Clusters by {selected_model}')
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        # Plot 2: Same points colored by fraud labels
        scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='RdYlBu', alpha=0.6)
        axes[1].set_title('True Fraud Labels')
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        # Add colorbars
        plt.colorbar(scatter1, ax=axes[0], label='Cluster')
        plt.colorbar(scatter2, ax=axes[1], label='Fraud (1=Fraud, 0=Legitimate)')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Cluster analysis summary
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        st.subheader("Cluster Analysis Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Number of Clusters", n_clusters)
        
        with col2:
            if -1 in labels:
                noise_count = np.sum(labels == -1)
                st.metric("Noise Points", noise_count)
            else:
                st.metric("Noise Points", 0)
        
        with col3:
            if n_clusters > 0:
                largest_cluster = max([np.sum(labels == i) for i in unique_labels if i != -1])
                st.metric("Largest Cluster Size", largest_cluster)

def _render_clustering_stability_analysis():
    """Render clustering stability analysis."""
    st.subheader("Clustering Stability Analysis")
    
    st.markdown("""
    ### Consistency Across Different Runs
    
    Test how stable clustering results are across multiple runs with different random seeds.
    """)
    
    unsupervised_models = get_unsupervised_models()
    selected_model = st.selectbox(
        "Select model for stability testing",
        options=list(unsupervised_models.keys()),
        key="stability_model"
    )
    
    n_runs = st.slider("Number of runs", 3, 20, 10)
    
    if st.button("Test Clustering Stability", key="stability_btn"):
        X, y = load_data(use_synthetic=True)
        
        # Preprocess data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        stability_results = []
        all_labels = []
        
        # Run clustering multiple times
        for i in range(n_runs):
            try:
                # Create a fresh model instance with different random state
                model_class = type(unsupervised_models[selected_model])
                if hasattr(unsupervised_models[selected_model], 'random_state'):
                    model = model_class(**{**unsupervised_models[selected_model].get_params(), 'random_state': i})
                else:
                    model = model_class(**unsupervised_models[selected_model].get_params())
                
                labels = model.fit_predict(X_scaled)
                all_labels.append(labels)
                
                # Calculate metrics for this run
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters > 1:
                    silhouette = silhouette_score(X_scaled, labels)
                else:
                    silhouette = np.nan
                
                stability_results.append({
                    'Run': i + 1,
                    'N_Clusters': n_clusters,
                    'Silhouette': silhouette
                })
                
            except Exception as e:
                st.warning(f"Error in run {i+1}: {str(e)}")
        
        if stability_results:
            results_df = pd.DataFrame(stability_results)
            
            # Calculate stability metrics
            cluster_counts = results_df['N_Clusters'].values
            silhouette_scores = results_df['Silhouette'].dropna().values
            
            # Visualize stability
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # Number of clusters variation
            axes[0].plot(results_df['Run'], results_df['N_Clusters'], 'bo-')
            axes[0].set_title('Number of Clusters Across Runs')
            axes[0].set_xlabel('Run')
            axes[0].set_ylabel('Number of Clusters')
            axes[0].grid(True, alpha=0.3)
            
            # Silhouette score variation
            valid_sil = results_df.dropna(subset=['Silhouette'])
            if not valid_sil.empty:
                axes[1].plot(valid_sil['Run'], valid_sil['Silhouette'], 'ro-')
                axes[1].set_title('Silhouette Score Across Runs')
                axes[1].set_xlabel('Run')
                axes[1].set_ylabel('Silhouette Score')
                axes[1].grid(True, alpha=0.3)
            
            # Distribution of cluster counts
            axes[2].hist(cluster_counts, bins=max(1, len(set(cluster_counts))), alpha=0.7, edgecolor='black')
            axes[2].set_title('Distribution of Cluster Counts')
            axes[2].set_xlabel('Number of Clusters')
            axes[2].set_ylabel('Frequency')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Stability summary
            st.subheader("Stability Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                cluster_std = np.std(cluster_counts)
                st.metric("Cluster Count Std", f"{cluster_std:.2f}")
            
            with col2:
                if len(silhouette_scores) > 0:
                    sil_std = np.std(silhouette_scores)
                    st.metric("Silhouette Std", f"{sil_std:.4f}")
                else:
                    st.metric("Silhouette Std", "N/A")
            
            with col3:
                most_common_clusters = max(set(cluster_counts), key=list(cluster_counts).count)
                st.metric("Most Common # Clusters", most_common_clusters)
            
            with col4:
                consistency_ratio = list(cluster_counts).count(most_common_clusters) / len(cluster_counts)
                st.metric("Consistency Ratio", f"{consistency_ratio:.2f}")
            
            # Stability interpretation
            if cluster_std < 0.5:
                st.success("✅ High stability - Consistent clustering across runs")
            elif cluster_std < 1.0:
                st.info("📊 Moderate stability - Some variation in results")
            else:
                st.warning("⚠️ Low stability - Results vary significantly across runs")

def _render_cross_model_comparison():
    """Render cross-model comparison between supervised and unsupervised."""
    st.subheader("Cross-Model Performance Comparison")
    
    st.markdown("""
    ### Supervised vs Unsupervised Performance
    
    Compare the effectiveness of supervised and unsupervised approaches for fraud detection.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Supervised Models")
        supervised_models = get_supervised_models()
        selected_supervised = st.multiselect(
            "Select supervised models",
            options=list(supervised_models.keys()),
            default=list(supervised_models.keys())[:2],
            key="cross_supervised"
        )
    
    with col2:
        st.markdown("#### Unsupervised Models")
        unsupervised_models = get_unsupervised_models()
        selected_unsupervised = st.multiselect(
            "Select unsupervised models",
            options=list(unsupervised_models.keys()),
            default=list(unsupervised_models.keys())[:2],
            key="cross_unsupervised"
        )
    
    if st.button("Compare Model Types", key="cross_compare_btn"):
        if selected_supervised or selected_unsupervised:
            X, y = load_data(use_synthetic=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_scaled = scaler.fit_transform(X)
            
            comparison_results = []
            
            # Evaluate supervised models
            for model_name in selected_supervised:
                model = supervised_models[model_name]
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                comparison_results.append({
                    'Model': model_name,
                    'Type': 'Supervised',
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred),
                    'Recall': recall_score(y_test, y_pred),
                    'F1_Score': f1_score(y_test, y_pred),
                    'Fraud_Detection_Rate': recall_score(y_test, y_pred)  # Same as recall for supervised
                })
            
            # Evaluate unsupervised models
            for model_name in selected_unsupervised:
                model = unsupervised_models[model_name]
                labels = model.fit_predict(X_scaled)
                
                # For unsupervised, we need to map clusters to fraud/legitimate
                # Use the cluster with highest fraud rate as "fraud cluster"
                unique_labels = set(labels)
                cluster_fraud_rates = {}
                
                for cluster_id in unique_labels:
                    if cluster_id != -1:
                        mask = labels == cluster_id
                        if mask.sum() > 0:
                            fraud_rate = y[mask].mean()
                            cluster_fraud_rates[cluster_id] = fraud_rate
                
                if cluster_fraud_rates:
                    # Assign the cluster with highest fraud rate as "fraud predictions"
                    fraud_cluster = max(cluster_fraud_rates.keys(), key=lambda k: cluster_fraud_rates[k])
                    y_pred_unsup = (labels == fraud_cluster).astype(int)
                    
                    # Calculate metrics
                    if len(set(y_pred_unsup)) > 1:  # Ensure we have both classes
                        accuracy = accuracy_score(y, y_pred_unsup)
                        precision = precision_score(y, y_pred_unsup, zero_division=0)
                        recall = recall_score(y, y_pred_unsup, zero_division=0)
                        f1 = f1_score(y, y_pred_unsup, zero_division=0)
                    else:
                        accuracy = precision = recall = f1 = 0
                    
                    fraud_detection_rate = max(cluster_fraud_rates.values())
                else:
                    accuracy = precision = recall = f1 = fraud_detection_rate = 0
                
                comparison_results.append({
                    'Model': model_name,
                    'Type': 'Unsupervised',
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1_Score': f1,
                    'Fraud_Detection_Rate': fraud_detection_rate
                })
            
            # Display results
            if comparison_results:
                results_df = pd.DataFrame(comparison_results)
                
                # Format for display
                display_df = results_df.copy()
                for col in ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'Fraud_Detection_Rate']:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
                
                st.dataframe(display_df, use_container_width=True)
                
                # Visualization
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
                
                for idx, metric in enumerate(metrics):
                    ax = axes[idx // 2, idx % 2]
                    
                    # Separate supervised and unsupervised results
                    sup_data = results_df[results_df['Type'] == 'Supervised']
                    unsup_data = results_df[results_df['Type'] == 'Unsupervised']
                    
                    x_pos = np.arange(len(results_df))
                    colors = ['blue' if t == 'Supervised' else 'red' for t in results_df['Type']]
                    
                    bars = ax.bar(x_pos, results_df[metric], color=colors, alpha=0.7)
                    ax.set_title(f'{metric} Comparison')
                    ax.set_ylabel(metric)
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(results_df['Model'], rotation=45)
                    
                    # Add legend
                    if idx == 0:
                        from matplotlib.patches import Patch
                        legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Supervised'),
                                         Patch(facecolor='red', alpha=0.7, label='Unsupervised')]
                        ax.legend(handles=legend_elements)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Summary insights
                st.subheader("Performance Insights")
                
                sup_results = results_df[results_df['Type'] == 'Supervised']
                unsup_results = results_df[results_df['Type'] == 'Unsupervised']
                
                if not sup_results.empty and not unsup_results.empty:
                    sup_avg_f1 = sup_results['F1_Score'].mean()
                    unsup_avg_f1 = unsup_results['F1_Score'].mean()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Avg Supervised F1", f"{sup_avg_f1:.4f}")
                    
                    with col2:
                        st.metric("Avg Unsupervised F1", f"{unsup_avg_f1:.4f}")
                    
                    with col3:
                        if sup_avg_f1 > unsup_avg_f1:
                            st.success("Supervised models perform better")
                        else:
                            st.info("Unsupervised models show promise")

def _render_confusion_matrix_analysis():
    """Render confusion matrix analysis."""
    st.subheader("Confusion Matrix Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Selection")
        supervised_models = get_supervised_models()
        selected_model = st.selectbox(
            "Choose model for analysis",
            options=list(supervised_models.keys()),
            key="conf_matrix_model"
        )
        
        if st.button("Generate Confusion Matrix Analysis", key="conf_matrix_btn"):
            with st.spinner("Analyzing confusion matrix..."):
                X, y = load_data(use_synthetic=True)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
                
                model = supervised_models[selected_model]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                cm = confusion_matrix(y_test, y_pred)
                
                # Display confusion matrix
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['Legitimate', 'Fraudulent'],
                           yticklabels=['Legitimate', 'Fraudulent'])
                ax.set_title(f'Confusion Matrix - {selected_model}')
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
                st.pyplot(fig)
                
                # Detailed metrics
                tn, fp, fn, tp = cm.ravel()
                
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("True Positives", tp)
                with col_b:
                    st.metric("False Positives", fp)
                with col_c:
                    st.metric("True Negatives", tn)
                with col_d:
                    st.metric("False Negatives", fn)
    
    with col2:
        st.markdown("### Analysis Insights")
        st.info("""
        **Confusion Matrix Interpretation:**
        
        - **True Positives (TP)**: Correctly identified fraud cases
        - **False Positives (FP)**: Legitimate addresses incorrectly flagged as fraud
        - **True Negatives (TN)**: Correctly identified legitimate addresses  
        - **False Negatives (FN)**: Fraud cases that were missed
        
        **Key Considerations:**
        - High FP rate impacts user experience
        - High FN rate means missing actual fraud
        - Balance depends on business priorities
        """)

def _render_roc_pr_analysis():
    """Render ROC and Precision-Recall analysis."""
    st.subheader("ROC & Precision-Recall Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        supervised_models = get_supervised_models()
        selected_models = st.multiselect(
            "Select models for ROC/PR analysis",
            options=list(supervised_models.keys()),
            default=list(supervised_models.keys())[:3],
            key="roc_pr_models"
        )
        
        if st.button("Generate ROC & PR Curves", key="roc_pr_btn"):
            if len(selected_models) > 0:
                X, y = load_data(use_synthetic=True)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                for model_name in selected_models:
                    model = supervised_models[model_name]
                    model.fit(X_train, y_train)
                    
                    if hasattr(model, 'predict_proba'):
                        y_prob = model.predict_proba(X_test)[:, 1]
                        
                        # ROC Curve
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        roc_auc = auc(fpr, tpr)
                        ax1.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
                        
                        # Precision-Recall Curve
                        precision, recall, _ = precision_recall_curve(y_test, y_prob)
                        avg_precision = average_precision_score(y_test, y_prob)
                        ax2.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.3f})')
                
                # ROC Plot formatting
                ax1.plot([0, 1], [0, 1], 'k--', label='Random')
                ax1.set_xlabel('False Positive Rate')
                ax1.set_ylabel('True Positive Rate')
                ax1.set_title('ROC Curves')
                ax1.legend()
                ax1.grid(True)
                
                # PR Plot formatting
                ax2.set_xlabel('Recall')
                ax2.set_ylabel('Precision')
                ax2.set_title('Precision-Recall Curves')
                ax2.legend()
                ax2.grid(True)
                
                plt.tight_layout()
                st.pyplot(fig)
    
    with col2:
        st.markdown("### Curve Interpretation")
        st.info("""
        **ROC Curve (Receiver Operating Characteristic):**
        - Shows trade-off between true positive rate and false positive rate
        - AUC closer to 1.0 indicates better performance
        - Good for balanced datasets
        
        **Precision-Recall Curve:**
        - Shows trade-off between precision and recall
        - Better for imbalanced datasets (like fraud detection)
        - Higher area under curve indicates better performance
        
        **When to use which:**
        - Use ROC when classes are roughly balanced
        - Use PR when positive class is rare (fraud detection)
        """)

def _render_threshold_optimization():
    """Render threshold optimization analysis."""
    st.subheader("Classification Threshold Optimization")
    
    st.markdown("""
    ### Finding the Optimal Decision Threshold
    
    The default threshold of 0.5 may not be optimal for fraud detection.
    This analysis helps find the best threshold based on business priorities.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        supervised_models = get_supervised_models()
        selected_model = st.selectbox(
            "Select model for threshold optimization",
            options=list(supervised_models.keys()),
            key="threshold_model"
        )
        
        optimization_metric = st.selectbox(
            "Optimization target",
            options=["F1-Score", "Precision", "Recall", "Balanced Accuracy"],
            key="threshold_metric"
        )
        
        if st.button("Optimize Threshold", key="threshold_btn"):
            X, y = load_data(use_synthetic=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            model = supervised_models[selected_model]
            model.fit(X_train, y_train)
            
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
                
                # Test different thresholds
                thresholds = np.arange(0.1, 0.9, 0.01)
                scores = []
                
                for threshold in thresholds:
                    y_pred_thresh = (y_prob >= threshold).astype(int)
                    
                    if optimization_metric == "F1-Score":
                        score = f1_score(y_test, y_pred_thresh)
                    elif optimization_metric == "Precision":
                        score = precision_score(y_test, y_pred_thresh)
                    elif optimization_metric == "Recall":
                        score = recall_score(y_test, y_pred_thresh)
                    else:  # Balanced Accuracy
                        score = balanced_accuracy_score(y_test, y_pred_thresh)
                    
                    scores.append(score)
                
                # Find optimal threshold
                optimal_idx = np.argmax(scores)
                optimal_threshold = thresholds[optimal_idx]
                optimal_score = scores[optimal_idx]
                
                # Plot threshold analysis
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(thresholds, scores, 'b-', linewidth=2)
                ax.axvline(x=optimal_threshold, color='r', linestyle='--', 
                          label=f'Optimal: {optimal_threshold:.3f}')
                ax.axvline(x=0.5, color='g', linestyle='--', alpha=0.7, 
                          label='Default: 0.5')
                ax.set_xlabel('Classification Threshold')
                ax.set_ylabel(optimization_metric)
                ax.set_title(f'{optimization_metric} vs Classification Threshold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Display results
                st.success(f"Optimal threshold: {optimal_threshold:.3f} with {optimization_metric}: {optimal_score:.3f}")
    
    with col2:
        st.markdown("### Threshold Guidelines")
        st.info("""
        **Threshold Selection Strategy:**
        
        **High Precision (Low FP):**
        - Use higher threshold (>0.7)
        - Reduces false alarms
        - May miss some fraud
        
        **High Recall (Low FN):**
        - Use lower threshold (<0.3)
        - Catches more fraud
        - More false alarms
        
        **Balanced Approach:**
        - Optimize F1-score
        - Balance precision and recall
        """)

def _render_cross_validation_analysis():
    """Render cross-validation analysis."""
    st.subheader("Cross-Validation Results")
    
    st.markdown("""
    ### Model Stability & Generalization Assessment
    
    Cross-validation helps assess how well models generalize to unseen data
    and provides insights into model stability across different data splits.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        supervised_models = get_supervised_models()
        selected_models = st.multiselect(
            "Select models for cross-validation",
            options=list(supervised_models.keys()),
            default=list(supervised_models.keys())[:3],
            key="cv_models"
        )
        
        cv_folds = st.slider("Number of CV folds", 3, 10, 5, key="cv_folds")
        
        if st.button("Run Cross-Validation Analysis", key="cv_btn"):
            if len(selected_models) > 0:
                X, y = load_data(use_synthetic=True)
                
                cv_results = {}
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                
                progress_bar = st.progress(0)
                
                for i, model_name in enumerate(selected_models):
                    model = supervised_models[model_name]
                    
                    # Multiple metrics
                    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
                    precision_scores = cross_val_score(model, X, y, cv=skf, scoring='precision')
                    recall_scores = cross_val_score(model, X, y, cv=skf, scoring='recall')
                    
                    cv_results[model_name] = {
                        'F1': f1_scores,
                        'Precision': precision_scores,
                        'Recall': recall_scores
                    }
                    
                    progress_bar.progress((i + 1) / len(selected_models))
                
                # Create results dataframe
                results_data = []
                for model_name, scores in cv_results.items():
                    for metric, values in scores.items():
                        results_data.append({
                            'Model': model_name,
                            'Metric': metric,
                            'Mean': values.mean(),
                            'Std': values.std(),
                            'Min': values.min(),
                            'Max': values.max()
                        })
                
                results_df = pd.DataFrame(results_data)
                
                # Display results table
                pivot_df = results_df.pivot(index='Model', columns='Metric', values='Mean')
                st.subheader("Cross-Validation Results (Mean Scores)")
                styled_cv = pivot_df.style.format('{:.4f}').background_gradient(cmap='Greens')
                st.dataframe(styled_cv)
                
                # Box plot of CV scores
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                metrics = ['F1', 'Precision', 'Recall']
                
                for idx, metric in enumerate(metrics):
                    data_to_plot = [cv_results[model][metric] for model in selected_models]
                    axes[idx].boxplot(data_to_plot, labels=selected_models)
                    axes[idx].set_title(f'{metric} Score Distribution')
                    axes[idx].tick_params(axis='x', rotation=45)
                    axes[idx].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
    
    with col2:
        st.markdown("### CV Interpretation")
        st.info("""
        **Cross-Validation Insights:**
        
        **Low Standard Deviation:**
        - Model is stable across folds
        - Good generalization
        
        **High Standard Deviation:**
        - Model is sensitive to data
        - May be overfitting
        
        **Consistent Performance:**
        - Model is reliable
        - Safe for production
        """)

def _render_performance_metrics_deep_dive():
    """Render detailed performance metrics analysis."""
    st.subheader("Performance Metrics Deep Dive")
    
    st.markdown("""
    ### Comprehensive Metric Analysis
    
    Deep dive into various performance metrics to understand model behavior
    from multiple perspectives and identify potential issues.
    """)
    
    # Allow selection of both supervised and unsupervised models
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Supervised Model Analysis")
        supervised_models = get_supervised_models()
        selected_supervised = st.selectbox(
            "Select supervised model for detailed analysis",
            options=list(supervised_models.keys()),
            key="metrics_supervised_model"
        )
        
        if st.button("Analyze Supervised Model", key="supervised_metrics_btn"):
            X, y = load_data(use_synthetic=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            model = supervised_models[selected_supervised]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate comprehensive metrics
            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'Matthews Correlation': matthews_corrcoef(y_test, y_pred),
                'Cohen Kappa': cohen_kappa_score(y_test, y_pred)
            }
            
            # Display metrics in organized layout
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("Classification Metrics")
                for metric, value in list(metrics.items())[:4]:
                    st.metric(metric, f"{value:.4f}")
            
            with col_b:
                st.subheader("Advanced Metrics")
                for metric, value in list(metrics.items())[4:]:
                    st.metric(metric, f"{value:.4f}")
    
    with col2:
        st.markdown("#### Unsupervised Model Analysis")
        unsupervised_models = get_unsupervised_models()
        selected_unsupervised = st.selectbox(
            "Select unsupervised model for detailed analysis",
            options=list(unsupervised_models.keys()),
            key="metrics_unsupervised_model"
        )
        
        if st.button("Analyze Unsupervised Model", key="unsupervised_metrics_btn"):
            X, y = load_data(use_synthetic=True)
            
            # Preprocess data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = unsupervised_models[selected_unsupervised]
            labels = model.fit_predict(X_scaled)
            
            # Calculate clustering metrics
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            metrics = {}
            
            if n_clusters > 1:
                metrics['Silhouette Score'] = silhouette_score(X_scaled, labels)
                try:
                    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
                    metrics['Calinski-Harabasz'] = calinski_harabasz_score(X_scaled, labels)
                    metrics['Davies-Bouldin'] = davies_bouldin_score(X_scaled, labels)
                except ImportError:
                    pass
            
            metrics['ARI Score'] = adjusted_rand_score(y, labels)
            metrics['Number of Clusters'] = n_clusters
            
            # Fraud detection effectiveness
            unique_labels = set(labels)
            cluster_fraud_rates = {}
            for cluster_id in unique_labels:
                if cluster_id != -1:
                    mask = labels == cluster_id
                    if mask.sum() > 0:
                        fraud_rate = y[mask].mean()
                        cluster_fraud_rates[cluster_id] = fraud_rate
            
            if cluster_fraud_rates:
                metrics['Max Fraud Rate'] = max(cluster_fraud_rates.values())
                metrics['Avg Fraud Rate'] = np.mean(list(cluster_fraud_rates.values()))
            
            # Display metrics
            st.subheader("Clustering Metrics")
            for metric, value in metrics.items():
                if isinstance(value, (int, np.integer)):
                    st.metric(metric, value)
                else:
                    st.metric(metric, f"{value:.4f}")
    
    # Metrics explanation
    st.subheader("Metric Interpretations")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("""
        **Supervised Metrics:**
        - **Accuracy**: Overall correctness
        - **Balanced Accuracy**: Accuracy adjusted for class imbalance
        - **Precision**: Of predicted fraud, how many are actually fraud
        - **Recall**: Of actual fraud, how many were caught
        - **F1-Score**: Harmonic mean of precision and recall
        - **Matthews Correlation**: Correlation between predictions and truth
        - **Cohen Kappa**: Agreement between predictions and truth
        """)
    
    with col_b:
        st.markdown("""
        **Unsupervised Metrics:**
        - **Silhouette Score**: Quality of cluster separation (-1 to 1)
        - **Calinski-Harabasz**: Ratio of between/within cluster variance
        - **Davies-Bouldin**: Average similarity between clusters (lower better)
        - **ARI Score**: Agreement with true labels (0 to 1)
        - **Max Fraud Rate**: Highest fraud concentration in any cluster
        - **Avg Fraud Rate**: Average fraud rate across all clusters
        """)
