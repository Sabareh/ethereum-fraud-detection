import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render_unsupervised_explainability(X, y):
    """Render the unsupervised model explainability page."""
    st.title("🔍 Cluster Analysis & Explainability")
    
    st.markdown("""
    ## Understanding Clustering Results
    
    This section provides comprehensive tools to understand and interpret clustering results,
    helping identify patterns in fraudulent vs legitimate transaction behaviors.
    """)
    
    # Check if clustering results exist
    if 'clustering_results' not in st.session_state:
        st.warning("⚠️ No clustering results found. Please run a clustering algorithm first in the 'Model Training' tab.")
        
        # Provide option to run quick clustering
        if st.button("🚀 Run Quick K-Means Analysis"):
            _run_quick_clustering(X, y)
        return
    
    results = st.session_state['clustering_results']
    labels = results['labels']
    algorithm = results['algorithm']
    X_2d = results['X_2d']
    n_clusters = results['n_clusters']
    
    # Create tabs for different analysis types
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Cluster Profiles", 
        "📊 Silhouette Analysis", 
        "⚠️ Fraud Distribution", 
        "🔍 Feature Analysis"
    ])
    
    with tab1:
        _render_cluster_profiles(X, labels, algorithm, n_clusters, y)
    
    with tab2:
        _render_silhouette_analysis(X, labels, algorithm, n_clusters)
    
    with tab3:
        _render_fraud_distribution_analysis(y, labels, n_clusters)
    
    with tab4:
        _render_feature_analysis(X, labels, n_clusters)

def _run_quick_clustering(X, y):
    """Run a quick K-Means clustering for demonstration."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    with st.spinner("Running K-Means clustering..."):
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Run K-Means
        kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Create 2D visualization
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X_scaled)
        
        # Store results
        st.session_state['clustering_results'] = {
            'model': kmeans,
            'algorithm': 'K-Means (Quick)',
            'labels': labels,
            'X_2d': X_2d,
            'n_clusters': 8,
            'cluster_fraud_rates': None
        }
        
        st.success("✅ Quick clustering completed! Refresh the page to see results.")
        st.rerun()

def _render_cluster_profiles(X, labels, algorithm, n_clusters, y):
    """Render cluster profiling analysis."""
    st.subheader(f"🎯 Cluster Profiles - {algorithm}")
    
    st.markdown("""
    ### Cluster Characteristics Summary
    
    Each cluster represents a group of addresses with similar transaction patterns.
    Understanding these profiles helps identify what makes certain clusters more prone to fraud.
    """)
    
    # Calculate cluster statistics
    cluster_stats = []
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    for cluster_id in range(n_clusters):
        if cluster_id in labels:
            mask = labels == cluster_id
            cluster_size = mask.sum()
            
            if cluster_size > 0:
                # Basic stats
                fraud_rate = y[mask].mean() if len(y) == len(labels) else 0
                avg_features = X[mask].mean(axis=0)
                
                cluster_stats.append({
                    'Cluster': cluster_id,
                    'Size': cluster_size,
                    'Fraud_Rate': fraud_rate,
                    'Risk_Level': _get_risk_level(fraud_rate),
                    'Percentage': (cluster_size / len(labels)) * 100
                })
    
    # Display cluster summary
    if cluster_stats:
        cluster_df = pd.DataFrame(cluster_stats)
        
        # Style the dataframe
        styled_df = cluster_df.style.format({
            'Fraud_Rate': '{:.2%}',
            'Percentage': '{:.1f}%'
        }).background_gradient(subset=['Fraud_Rate'], cmap='Reds')
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Cluster size visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Cluster Size Distribution")
            fig_size = px.pie(
                cluster_df, 
                values='Size', 
                names='Cluster',
                title="Distribution of Addresses Across Clusters"
            )
            st.plotly_chart(fig_size, use_container_width=True)
        
        with col2:
            st.subheader("⚠️ Fraud Risk by Cluster")
            fig_risk = px.bar(
                cluster_df,
                x='Cluster',
                y='Fraud_Rate',
                color='Risk_Level',
                title="Fraud Rate by Cluster",
                color_discrete_map={
                    'Low Risk': 'green',
                    'Medium Risk': 'orange', 
                    'High Risk': 'red'
                }
            )
            fig_risk.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig_risk, use_container_width=True)

def _render_silhouette_analysis(X, labels, algorithm, n_clusters):
    """Render silhouette analysis."""
    st.subheader(f"📊 Silhouette Analysis - {algorithm}")
    
    st.markdown("""
    ### Cluster Quality Assessment
    
    Silhouette analysis measures how well-separated clusters are:
    - **Score > 0.7**: Excellent separation
    - **Score 0.5-0.7**: Good separation  
    - **Score 0.3-0.5**: Weak separation
    - **Score < 0.3**: Poor separation
    """)
    
    try:
        # Calculate silhouette scores
        if n_clusters > 1:
            sample_silhouette_values = silhouette_samples(X, labels)
            silhouette_avg = silhouette_score(X, labels)
            
            # Display overall score
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Silhouette Score", f"{silhouette_avg:.3f}")
            with col2:
                quality = "Excellent" if silhouette_avg > 0.7 else "Good" if silhouette_avg > 0.5 else "Weak" if silhouette_avg > 0.3 else "Poor"
                st.metric("Quality Assessment", quality)
            with col3:
                st.metric("Number of Clusters", n_clusters)
            
            # Silhouette plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            y_lower = 10
            for cluster_id in range(n_clusters):
                if cluster_id in labels:
                    cluster_silhouette_values = sample_silhouette_values[labels == cluster_id]
                    cluster_silhouette_values.sort()
                    
                    size_cluster_i = cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i
                    
                    color = plt.cm.nipy_spectral(float(cluster_id) / n_clusters)
                    ax.fill_betweenx(np.arange(y_lower, y_upper),
                                    0, cluster_silhouette_values,
                                    facecolor=color, edgecolor=color, alpha=0.7)
                    
                    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster_id))
                    y_lower = y_upper + 10
            
            ax.set_xlabel('Silhouette Coefficient Values')
            ax.set_ylabel('Cluster Label')
            ax.set_title('Silhouette Plot for Clusters')
            
            # Add average line
            ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
                      label=f'Average Score: {silhouette_avg:.3f}')
            ax.legend()
            
            st.pyplot(fig)
            
            # Interpretation
            st.markdown("""
            ### 📝 Interpretation Guide
            
            - **Width of each cluster**: Indicates cluster size
            - **Thickness above average line**: Well-separated cluster
            - **Thickness below average line**: Poorly separated cluster
            - **Uniform thickness**: Well-balanced clustering
            """)
            
        else:
            st.warning("Silhouette analysis requires at least 2 clusters.")
    
    except Exception as e:
        st.error(f"Error in silhouette analysis: {str(e)}")

def _render_fraud_distribution_analysis(y, labels, n_clusters):
    """Render fraud distribution analysis."""
    st.subheader("⚠️ Fraud Distribution Analysis")
    
    st.markdown("""
    ### How Well Does Clustering Identify Fraud?
    
    This analysis shows how fraudulent addresses are distributed across clusters,
    helping identify which clusters are most suspicious.
    """)
    
    if len(y) != len(labels):
        st.warning("Fraud labels not available for this clustering analysis.")
        return
    
    # Calculate fraud distribution
    fraud_analysis = []
    total_fraud = y.sum()
    total_legitimate = len(y) - total_fraud
    
    for cluster_id in range(n_clusters):
        if cluster_id in labels:
            mask = labels == cluster_id
            cluster_size = mask.sum()
            
            if cluster_size > 0:
                cluster_fraud = y[mask].sum()
                cluster_legitimate = cluster_size - cluster_fraud
                fraud_rate = cluster_fraud / cluster_size
                fraud_concentration = cluster_fraud / total_fraud if total_fraud > 0 else 0
                
                fraud_analysis.append({
                    'Cluster': cluster_id,
                    'Total_Addresses': cluster_size,
                    'Fraudulent': cluster_fraud,
                    'Legitimate': cluster_legitimate,
                    'Fraud_Rate': fraud_rate,
                    'Fraud_Concentration': fraud_concentration,
                    'Risk_Level': _get_risk_level(fraud_rate)
                })
    
    if fraud_analysis:
        fraud_df = pd.DataFrame(fraud_analysis)
        
        # Display fraud analysis table
        styled_fraud_df = fraud_df.style.format({
            'Fraud_Rate': '{:.2%}',
            'Fraud_Concentration': '{:.2%}'
        }).background_gradient(subset=['Fraud_Rate'], cmap='Reds')
        
        st.dataframe(styled_fraud_df, use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Fraud Concentration by Cluster")
            fig_conc = px.bar(
                fraud_df,
                x='Cluster',
                y='Fraud_Concentration',
                color='Risk_Level',
                title="Percentage of Total Fraud in Each Cluster",
                color_discrete_map={
                    'Low Risk': 'green',
                    'Medium Risk': 'orange', 
                    'High Risk': 'red'
                }
            )
            fig_conc.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig_conc, use_container_width=True)
        
        with col2:
            st.subheader("📊 Cluster Composition")
            
            # Create stacked bar chart
            fig_comp = go.Figure(data=[
                go.Bar(name='Legitimate', x=fraud_df['Cluster'], y=fraud_df['Legitimate']),
                go.Bar(name='Fraudulent', x=fraud_df['Cluster'], y=fraud_df['Fraudulent'])
            ])
            
            fig_comp.update_layout(
                barmode='stack',
                title='Address Composition by Cluster',
                xaxis_title='Cluster',
                yaxis_title='Number of Addresses'
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        
        # Insights
        high_risk_clusters = fraud_df[fraud_df['Risk_Level'] == 'High Risk']['Cluster'].tolist()
        medium_risk_clusters = fraud_df[fraud_df['Risk_Level'] == 'Medium Risk']['Cluster'].tolist()
        
        if high_risk_clusters:
            st.error(f"🚨 **High Risk Clusters**: {high_risk_clusters} - These clusters have high fraud concentrations and should be prioritized for investigation.")
        
        if medium_risk_clusters:
            st.warning(f"⚠️ **Medium Risk Clusters**: {medium_risk_clusters} - These clusters show elevated fraud rates and warrant monitoring.")
        
        # Overall clustering effectiveness
        max_fraud_rate = fraud_df['Fraud_Rate'].max()
        best_cluster = fraud_df.loc[fraud_df['Fraud_Rate'].idxmax(), 'Cluster']
        
        st.info(f"""
        **Clustering Effectiveness Summary:**
        
        🎯 **Best Fraud Detection**: Cluster {best_cluster} with {max_fraud_rate:.1%} fraud rate
        
        📊 **Overall Assessment**: {'Excellent' if max_fraud_rate > 0.8 else 'Good' if max_fraud_rate > 0.6 else 'Moderate' if max_fraud_rate > 0.4 else 'Poor'} fraud isolation capability
        
        💡 **Recommendation**: {'Focus investigative resources on high-risk clusters' if high_risk_clusters else 'Monitor medium-risk clusters closely' if medium_risk_clusters else 'Consider alternative clustering approaches'}
        """)

def _render_feature_analysis(X, labels, n_clusters):
    """Render feature analysis for clusters."""
    st.subheader("🔍 Feature Analysis by Cluster")
    
    st.markdown("""
    ### Understanding What Differentiates Clusters
    
    This analysis shows which features drive cluster formation and help distinguish
    between different types of address behaviors.
    """)
    
    feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    # Calculate feature means per cluster
    cluster_features = []
    
    for cluster_id in range(n_clusters):
        if cluster_id in labels:
            mask = labels == cluster_id
            if mask.sum() > 0:
                cluster_means = X[mask].mean(axis=0)
                cluster_features.append({
                    'Cluster': cluster_id,
                    **{f'Feature_{i}': cluster_means[i] for i in range(len(cluster_means))}
                })
    
    if cluster_features:
        cluster_feature_df = pd.DataFrame(cluster_features)
        
        # Feature selection for detailed analysis
        selected_features = st.multiselect(
            "Select features to analyze",
            options=feature_names,
            default=feature_names[:5] if len(feature_names) >= 5 else feature_names,
            help="Choose which features to examine across clusters"
        )
        
        if selected_features:
            # Create feature comparison heatmap
            st.subheader("🌡️ Feature Heatmap by Cluster")
            
            heatmap_data = cluster_feature_df[['Cluster'] + selected_features].set_index('Cluster')
            
            fig_heatmap = px.imshow(
                heatmap_data.T,
                labels=dict(x="Cluster", y="Feature", color="Value"),
                title="Feature Values Across Clusters",
                aspect="auto"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Feature variance analysis
            st.subheader("📊 Feature Variance Across Clusters")
            
            feature_variance = {}
            for feature in selected_features:
                variance = cluster_feature_df[feature].var()
                feature_variance[feature] = variance
            
            variance_df = pd.DataFrame(
                list(feature_variance.items()), 
                columns=['Feature', 'Variance']
            ).sort_values('Variance', ascending=False)
            
            fig_var = px.bar(
                variance_df,
                x='Feature',
                y='Variance',
                title="Feature Variance Across Clusters (Higher = More Discriminative)"
            )
            fig_var.update_xaxes(tickangle=45)
            st.plotly_chart(fig_var, use_container_width=True)
            
            # Most discriminative features
            top_discriminative = variance_df.head(3)['Feature'].tolist()
            st.success(f"🎯 **Most Discriminative Features**: {', '.join(top_discriminative)}")
            st.info("These features show the highest variance across clusters, making them key drivers of cluster formation.")

def _get_risk_level(fraud_rate):
    """Convert fraud rate to risk level."""
    if fraud_rate >= 0.7:
        return "High Risk"
    elif fraud_rate >= 0.3:
        return "Medium Risk"
    else:
        return "Low Risk"
