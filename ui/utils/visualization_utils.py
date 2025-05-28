import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st

def create_correlation_heatmap(corr, mask):
    """Create correlation heatmap visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    return fig

def create_transaction_histogram(df, transaction_col):
    """Create transaction volume histogram."""
    fig, ax = plt.subplots()
    try:
        sns.histplot(data=df, x=transaction_col, bins=50, kde=True, ax=ax)
        ax.set_xlabel(f"Number of Transactions ({transaction_col})")
        ax.set_ylabel("Frequency")
        return fig
    except Exception as e:
        st.error(f"Error creating transaction histogram: {str(e)}")
        return None

def create_pca_plot(pca_results, labels, title="PCA Projection of Transaction Features"):
    """Create PCA 2D projection plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(pca_results[:, 0], pca_results[:, 1], c=labels, cmap='coolwarm', s=10, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    legend = ax.legend(*scatter.legend_elements(), title="Class")
    ax.add_artist(legend)
    return fig

def create_tsne_plot(tsne_results, labels, title="t-SNE Projection of Transaction Features"):
    """Create t-SNE 2D projection plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='coolwarm', s=10, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    legend = ax.legend(*scatter.legend_elements(), title="Class")
    ax.add_artist(legend)
    return fig

def create_clustering_visualization(X_2d, labels, algorithm_name):
    """Create clustering results visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each cluster
    unique_labels = set(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise
            col = 'black'
            marker = 'x'
            label = 'Noise'
        else:
            marker = 'o'
            label = f'Cluster {k}'
        
        class_member_mask = (labels == k)
        xy = X_2d[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, 
                 s=50, alpha=0.6, label=label)
    
    ax.set_title(f'{algorithm_name} Clustering Results')
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.legend()
    return fig

def create_unsupervised_comparison_plots(results_df):
    """Create comparison plots for unsupervised models."""
    plots = {}
    
    # Filter out rows with 'N/A' values for visualization
    viz_data = results_df[results_df['Silhouette'] != 'N/A'].copy()
    
    # Silhouette score comparison
    if not viz_data.empty:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        bars = ax1.bar(viz_data['Model'], pd.to_numeric(viz_data['Silhouette']), color='skyblue')
        ax1.set_xlabel('Clustering Algorithm')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title('Cluster Quality Comparison')
        ax1.set_ylim(0, max(pd.to_numeric(viz_data['Silhouette'])) * 1.2)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.2f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plots['silhouette'] = fig1
    
    # Runtime comparison
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    runtime_data = results_df.copy()
    bars2 = ax2.bar(runtime_data['Model'], pd.to_numeric(runtime_data['Runtime(s)']), color='lightgreen')
    ax2.set_xlabel('Clustering Algorithm')
    ax2.set_ylabel('Runtime (seconds)')
    ax2.set_title('Computational Efficiency Comparison')
    ax2.set_ylim(0, max(pd.to_numeric(runtime_data['Runtime(s)'])) * 1.2)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{height:.2f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plots['runtime'] = fig2
    
    # Multi-metric comparison (bubble chart)
    complete_data = results_df[results_df['Silhouette'] != 'N/A'].copy()
    
    if not complete_data.empty:
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        
        # Convert columns to numeric
        complete_data['Silhouette'] = pd.to_numeric(complete_data['Silhouette'])
        complete_data['Runtime(s)'] = pd.to_numeric(complete_data['Runtime(s)'])
        complete_data['Fraud_Detection'] = pd.to_numeric(complete_data['Fraud_Detection'])
        complete_data['Clusters'] = pd.to_numeric(complete_data['Clusters'])
        
        scatter = ax3.scatter(
            complete_data['Silhouette'], 
            complete_data['Fraud_Detection'],
            s=100 / complete_data['Runtime(s)'],
            c=complete_data['Clusters'],
            cmap='viridis',
            alpha=0.7
        )
        
        # Add labels for each point
        for i, model in enumerate(complete_data['Model']):
            ax3.annotate(model, 
                       (complete_data['Silhouette'].iloc[i], complete_data['Fraud_Detection'].iloc[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.set_xlabel('Silhouette Score (cluster quality)')
        ax3.set_ylabel('Fraud Detection Rate')
        ax3.set_title('Multi-dimensional Algorithm Comparison')
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Add colorbar for cluster count
        cbar = plt.colorbar(scatter)
        cbar.set_label('Number of Clusters')
        
        # Add annotation for bubble size
        ax3.text(0.05, 0.05, 'Bubble size inversely proportional to runtime (larger = faster)', 
               transform=ax3.transAxes, fontsize=10)
        
        plt.tight_layout()
        plots['multimetric'] = fig3
    
    return plots

def create_roc_curve(y_test, y_proba):
    """Create ROC curve plot."""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    return fig

def create_feature_importance_plot(importance_df, top_n=15):
    """Create feature importance plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = importance_df.head(top_n)
    ax.barh(range(len(top_features)), top_features['importance'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_n} Most Important Features')
    return fig
