import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from model_utils import get_supervised_models, get_unsupervised_models, load_data

def render_model_analysis_page():
    """Main function to render the comprehensive model analysis page."""
    st.title("🔬 Advanced Model Analysis")
    
    st.markdown("""
    ### Comprehensive Model Evaluation & Analysis Suite
    
    This advanced analysis suite provides deep insights into model performance, 
    behavior, and characteristics using both supervised and unsupervised approaches.
    """)
    
    # Main analysis tabs
    analysis_tabs = st.tabs([
        "🔬 Unified Analysis",
        "🎯 Performance Analysis", 
        "📈 Learning Curves",
        "⚡ Efficiency Analysis",
        "🔍 Robustness Testing"
    ])
    
    with analysis_tabs[0]:
        try:
            from pages.analysis.unified_analysis import render_unified_analysis_tab
            render_unified_analysis_tab()
        except Exception as e:
            st.error(f"Error in Unified Analysis: {str(e)}")
            _render_fallback_unified_analysis()
    
    with analysis_tabs[1]:
        try:
            from pages.analysis.performance_analysis import render_performance_analysis_tab
            render_performance_analysis_tab()
        except Exception as e:
            st.error(f"Error in Performance Analysis: {str(e)}")
            _render_fallback_performance_analysis()
    
    with analysis_tabs[2]:
        try:
            from pages.analysis.learning_curves import render_learning_curves_tab
            render_learning_curves_tab()
        except Exception as e:
            st.error(f"Error in Learning Curves: {str(e)}")
            _render_fallback_learning_curves()
    
    with analysis_tabs[3]:
        try:
            from pages.analysis.efficiency_analysis import render_efficiency_analysis_tab
            render_efficiency_analysis_tab()
        except Exception as e:
            st.error(f"Error in Efficiency Analysis: {str(e)}")
            _render_fallback_efficiency_analysis()
    
    with analysis_tabs[4]:
        try:
            from pages.analysis.robustness_testing import render_robustness_testing_tab
            render_robustness_testing_tab()
        except Exception as e:
            st.error(f"Error in Robustness Testing: {str(e)}")
            _render_fallback_robustness_testing()
    
    # Quick analysis summary
    st.markdown("---")
    st.subheader("📊 Quick Analysis Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Analysis Modules", 
            "5",
            help="Number of analysis modules available"
        )
    
    with col2:
        st.metric(
            "Model Types Supported", 
            "Both",
            help="Supports supervised and unsupervised models"
        )
    
    with col3:
        st.metric(
            "Analysis Categories", 
            "15+",
            help="Over 15 different analysis categories"
        )
    
    with col4:
        st.metric(
            "Visualization Types", 
            "20+",
            help="Multiple visualization options"
        )
    
    # Analysis guidance
    with st.expander("📖 Analysis Guide"):
        st.markdown("""
        ### How to Use This Analysis Suite
        
        **🔬 Unified Analysis:**
        - Comprehensive unsupervised ML analysis
        - Clustering, dimensionality reduction, anomaly detection
        - Pattern discovery and feature analysis
        
        **🎯 Performance Analysis:**
        - Detailed performance metrics
        - ROC curves, confusion matrices
        - Cross-validation results
        
        **📈 Learning Curves:**
        - Training vs validation performance
        - Sample size impact analysis
        - Hyperparameter optimization
        
        **⚡ Efficiency Analysis:**
        - Training and inference speed
        - Memory usage analysis
        - Scalability testing
        
        **🔍 Robustness Testing:**
        - Adversarial testing
        - Data perturbation analysis
        - Edge case evaluation
        
        ### Best Practices:
        1. Start with Unified Analysis for overview
        2. Use Performance Analysis for detailed metrics
        3. Check Learning Curves for optimization opportunities
        4. Review Efficiency for deployment planning
        5. Validate with Robustness Testing
        """)

# Fallback functions when modules are not available
def _render_fallback_unified_analysis():
    """Fallback unified analysis when module is not available."""
    st.subheader("🔬 Basic Unified Analysis")
    
    st.markdown("""
    ### Model Comparison Dashboard
    
    Compare supervised and unsupervised models for fraud detection.
    """)
    
    # Model type selection
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Supervised Models", "Unsupervised Models", "Combined Analysis"],
        help="Choose which type of models to analyze"
    )
    
    if analysis_type == "Supervised Models":
        _render_basic_supervised_comparison()
    elif analysis_type == "Unsupervised Models":
        _render_basic_unsupervised_comparison()
    else:
        _render_basic_combined_analysis()

def _render_basic_supervised_comparison():
    """Basic supervised model comparison."""
    st.subheader("📊 Supervised Model Comparison")
    
    supervised_models = get_supervised_models()
    selected_models = st.multiselect(
        "Choose models for comparison",
        options=list(supervised_models.keys()),
        default=list(supervised_models.keys())[:3],
        help="Select 2-6 models for meaningful comparison"
    )
    
    if len(selected_models) < 2:
        st.warning("Please select at least 2 models for comparison.")
        return
    
    if st.button("🚀 Run Supervised Model Comparison", type="primary"):
        X, y = load_data(use_synthetic=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        results = []
        progress_bar = st.progress(0)
        
        for i, model_name in enumerate(selected_models):
            model = clone(supervised_models[model_name])
            
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            y_pred = model.predict(X_test)
            
            results.append({
                'Model': model_name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1_Score': f1_score(y_test, y_pred),
                'Training_Time': training_time
            })
            
            progress_bar.progress((i + 1) / len(selected_models))
        
        # Display results
        results_df = pd.DataFrame(results)
        st.dataframe(results_df.style.format({
            'Accuracy': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1_Score': '{:.4f}',
            'Training_Time': '{:.3f}s'
        }).background_gradient(subset=['F1_Score'], cmap='Greens'), use_container_width=True)

def _render_basic_unsupervised_comparison():
    """Basic unsupervised model comparison."""
    st.subheader("🔍 Unsupervised Model Comparison")
    
    unsupervised_models = get_unsupervised_models()
    selected_models = st.multiselect(
        "Choose models for comparison",
        options=list(unsupervised_models.keys()),
        default=list(unsupervised_models.keys())[:3],
        help="Select models for clustering comparison"
    )
    
    if len(selected_models) < 2:
        st.warning("Please select at least 2 models for comparison.")
        return
    
    if st.button("🔍 Run Unsupervised Model Comparison", type="primary"):
        X, y = load_data(use_synthetic=True)
        
        # Preprocess data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        results = []
        progress_bar = st.progress(0)
        
        for i, model_name in enumerate(selected_models):
            try:
                model = clone(unsupervised_models[model_name])
                
                start_time = time.time()
                labels = model.fit_predict(X_scaled)
                training_time = time.time() - start_time
                
                # Calculate metrics
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                if n_clusters > 1:
                    sil_score = silhouette_score(X_scaled, labels)
                else:
                    sil_score = np.nan
                
                # Calculate fraud detection effectiveness
                fraud_rates = []
                for cluster_id in set(labels):
                    if cluster_id != -1:
                        mask = labels == cluster_id
                        if mask.sum() > 0:
                            fraud_rate = y[mask].mean()
                            fraud_rates.append(fraud_rate)
                
                max_fraud_rate = max(fraud_rates) if fraud_rates else 0
                
                ari_score = adjusted_rand_score(y, labels)
                
                results.append({
                    'Model': model_name,
                    'N_Clusters': n_clusters,
                    'Silhouette_Score': sil_score,
                    'Max_Fraud_Rate': max_fraud_rate,
                    'ARI_Score': ari_score,
                    'Training_Time': training_time
                })
                
            except Exception as e:
                st.error(f"Error running {model_name}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(selected_models))
        
        # Display results
        if results:
            results_df = pd.DataFrame(results)
            display_df = results_df.copy()
            
            # Format numeric columns
            for col in ['Silhouette_Score', 'Max_Fraud_Rate', 'ARI_Score']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
            
            display_df['Training_Time'] = display_df['Training_Time'].apply(lambda x: f"{x:.4f}s")
            
            st.dataframe(display_df, use_container_width=True)

def _render_basic_combined_analysis():
    """Basic combined analysis."""
    st.subheader("🔄 Combined Model Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        supervised_models = get_supervised_models()
        selected_supervised = st.selectbox(
            "Select Supervised Model",
            options=list(supervised_models.keys()),
            help="Choose one supervised model for comparison"
        )
    
    with col2:
        unsupervised_models = get_unsupervised_models()
        selected_unsupervised = st.selectbox(
            "Select Unsupervised Model",
            options=list(unsupervised_models.keys()),
            help="Choose one unsupervised model for comparison"
        )
    
    if st.button("🔗 Run Combined Analysis", type="primary"):
        X, y = load_data(use_synthetic=True)
        
        # Preprocess data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
        
        # Run supervised model
        sup_model = clone(supervised_models[selected_supervised])
        sup_model.fit(X_train, y_train)
        sup_pred = sup_model.predict(X_test)
        
        from sklearn.metrics import accuracy_score, f1_score
        
        supervised_results = {
            'accuracy': accuracy_score(y_test, sup_pred),
            'f1': f1_score(y_test, sup_pred)
        }
        
        # Run unsupervised model
        unsup_model = clone(unsupervised_models[selected_unsupervised])
        unsup_labels = unsup_model.fit_predict(X_scaled)
        
        n_clusters = len(set(unsup_labels)) - (1 if -1 in unsup_labels else 0)
        
        if n_clusters > 1:
            sil_score = silhouette_score(X_scaled, unsup_labels)
        else:
            sil_score = np.nan
        
        # Calculate fraud detection effectiveness
        fraud_rates = []
        for cluster_id in set(unsup_labels):
            if cluster_id != -1:
                mask = unsup_labels == cluster_id
                if mask.sum() > 0:
                    fraud_rate = y[mask].mean()
                    fraud_rates.append(fraud_rate)
        
        max_fraud_rate = max(fraud_rates) if fraud_rates else 0
        
        unsupervised_results = {
            'n_clusters': n_clusters,
            'silhouette': sil_score,
            'fraud_detection': max_fraud_rate
        }
        
        # Display results
        st.subheader("🔗 Combined Analysis Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Supervised Model: {selected_supervised}**")
            st.metric("Accuracy", f"{supervised_results['accuracy']:.4f}")
            st.metric("F1-Score", f"{supervised_results['f1']:.4f}")
        
        with col2:
            st.markdown(f"**Unsupervised Model: {selected_unsupervised}**")
            st.metric("Clusters", unsupervised_results['n_clusters'])
            st.metric("Silhouette Score", f"{unsupervised_results['silhouette']:.4f}" if not pd.isna(unsupervised_results['silhouette']) else "N/A")
            st.metric("Max Fraud Rate", f"{unsupervised_results['fraud_detection']:.4f}")

def _render_fallback_performance_analysis():
    """Fallback performance analysis."""
    st.info("📈 Performance Analysis module is being loaded. Please check back later or refer to the basic comparison above.")

def _render_fallback_learning_curves():
    """Fallback learning curves analysis."""
    st.info("📊 Learning Curves module is being loaded. This will show training vs validation performance curves.")

def _render_fallback_efficiency_analysis():
    """Fallback efficiency analysis."""
    st.info("⚡ Efficiency Analysis module is being loaded. This will show computational performance metrics.")

def _render_fallback_robustness_testing():
    """Fallback robustness testing."""
    st.info("🔍 Robustness Testing module is being loaded. This will test model stability under various conditions.")

# Ensure the main function is available at module level
__all__ = ['render_model_analysis_page']
