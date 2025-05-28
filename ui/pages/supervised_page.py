import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from model_utils import get_supervised_models
from interpretability_utils import ModelExplainer, compare_feature_importance_methods, create_decision_boundary_plot
from utils.visualization_utils import create_roc_curve, create_feature_importance_plot

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def render_supervised_page(X, y):
    """Render the supervised learning page."""
    st.title("Supervised Classification")
    
    try:
        models = get_supervised_models()
        
        # Add tabs for different views
        tab_names = ["Model Training", "Model Comparison", "Model Explainability"]
        tabs = st.tabs(tab_names)
        
        with tabs[0]:
            _render_model_training_tab(models, X, y)
        
        with tabs[1]:
            _render_model_comparison_tab()
        
        with tabs[2]:
            _render_model_explainability_tab(models, X, y)
    
    except Exception as e:
        st.error(f"Error in supervised modeling: {str(e)}")

def _render_model_training_tab(models, X, y):
    """Render the model training tab."""
    available_models = list(models.keys())
    if not available_models:
        st.warning("No supervised models available. Please check library installations.")
        return
        
    alg = st.selectbox("Algorithm", available_models)
    model = models[alg]
    test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.3)
    
    with st.expander("Train and evaluate model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=42, stratify=y)
        with st.spinner(f"Training {alg}..."):
            model.fit(X_train, y_train)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model Performance")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            st.metric("Accuracy", f"{accuracy:.3f}")
            st.metric("Precision", f"{precision:.3f}")
            st.metric("Recall", f"{recall:.3f}")
            st.metric("F1-Score", f"{f1:.3f}")
        
        with col2:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.json(report)
        
        # Add ROC curve if applicable
        if hasattr(model, "predict_proba"):
            st.subheader("ROC Curve")
            y_proba = model.predict_proba(X_test)[:, 1]
            fig = create_roc_curve(y_test, y_proba)
            st.pyplot(fig)
        
        # Feature importance if available
        if hasattr(model, "feature_importances_"):
            st.subheader("Feature Importance")
            feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig = create_feature_importance_plot(importance_df, top_n=15)
            st.pyplot(fig)

def _render_model_comparison_tab():
    """Render the model comparison tab."""
    st.subheader("Supervised Model Comparison")
    
    # Attempt to load detailed comparison documentation
    try:
        sup_doc_path = os.path.join(BASE_DIR, "docs", "supervised_model_comparison.md")
        if os.path.exists(sup_doc_path):
            with open(sup_doc_path, "r", encoding="utf-8") as f:
                sup_doc = f.read()
            st.markdown(sup_doc)
        else:
            st.warning("Detailed supervised model comparison documentation not found.")
            _render_fallback_supervised_comparison()
    except Exception as e:
        st.error(f"Error loading model comparison: {str(e)}")
        _render_fallback_supervised_comparison()

def _render_fallback_supervised_comparison():
    """Render fallback comparison when documentation is not available."""
    st.markdown("""
    ## Supervised Model Performance Comparison
    
    Our analysis evaluated multiple supervised learning algorithms for Ethereum fraud detection.
    The models were tested on the same dataset with consistent preprocessing and evaluation metrics.
    
    ### Key Findings:
    
    1. **Gradient Boosting Methods** (XGBoost, LightGBM) showed superior performance
    2. **Ensemble Methods** provided robust predictions with high accuracy
    3. **Neural Networks** (TabNet, MLP) showed promise but required more tuning
    4. **Traditional ML** methods (SVM, Random Forest) provided good baseline performance
    
    ### Performance Summary:
    - **Best Accuracy**: XGBoost (93.9%)
    - **Best Speed**: LightGBM (0.12s training time)
    - **Best Balance**: LightGBM (93.8% accuracy, fast training)
    - **Most Robust**: Stacking Ensemble (93.9% accuracy)
    """)

def _render_model_explainability_tab(models, X, y):
    """Render the model explainability tab."""
    st.subheader("Model Explainability & Interpretability")
    
    st.markdown("""
    ### Understanding Model Decisions
    
    Model explainability is crucial for fraud detection systems, especially in regulated environments
    where decisions need to be justified. This section provides comprehensive tools to understand
    how our models make predictions and which features drive their decisions.
    
    **Available Interpretation Methods:**
    - **Feature Importance**: Multiple methods to rank feature significance
    - **SHAP Analysis**: Game-theoretic approach to explain individual predictions
    - **LIME**: Local interpretable model-agnostic explanations
    - **Partial Dependence**: How individual features affect predictions
    - **Instance-level Explanations**: Detailed breakdown of specific predictions
    """)
    
    # Model selection for explainability
    model_choice = st.selectbox(
        "Select Model for Explanation",
        options=list(models.keys()),
        index=0,
        help="Choose which model to analyze for explainability"
    )
    
    if st.button("Initialize Explainer", help="Load the selected model and prepare explainability tools"):
        with st.spinner("Training model and initializing explainer..."):
            try:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
                
                # Train selected model
                selected_model = models[model_choice]
                selected_model.fit(X_train, y_train)
                
                # Create feature names (simplified for demo)
                feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
                
                # Initialize explainer
                explainer = ModelExplainer(
                    model=selected_model,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    feature_names=feature_names
                )
                
                # Store in session state
                st.session_state['explainer'] = explainer
                st.session_state['selected_model'] = selected_model
                st.session_state['model_name'] = model_choice
                
                st.success(f"✅ Explainer initialized for {model_choice}")
                
            except Exception as e:
                st.error(f"Error initializing explainer: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Show explainability tools if explainer is available
    if 'explainer' in st.session_state:
        _render_explainability_tools()
    else:
        st.info("👆 Please initialize the explainer first by selecting a model and clicking 'Initialize Explainer'")

def _render_explainability_tools():
    """Render the explainability tools interface."""
    explainer = st.session_state['explainer']
    model_name = st.session_state['model_name']
    
    st.success(f"Explainer ready for {model_name}")
    
    # Create explainability sub-tabs
    exp_tabs = st.tabs([
        "Feature Importance", 
        "SHAP Analysis", 
        "Instance Explanations", 
        "Partial Dependence",
        "Model Analysis"
    ])
    
    with exp_tabs[0]:
        _render_feature_importance_tab(explainer)
    
    with exp_tabs[1]:
        _render_shap_analysis_tab(explainer)
    
    with exp_tabs[2]:
        _render_instance_explanations_tab(explainer)
    
    with exp_tabs[3]:
        _render_partial_dependence_tab(explainer)
    
    with exp_tabs[4]:
        _render_model_analysis_tab(explainer)

def _render_feature_importance_tab(explainer):
    """Render feature importance analysis."""
    st.markdown("### Feature Importance Analysis")
    st.markdown("""
    Feature importance helps us understand which transaction characteristics
    are most predictive of fraud. We compare multiple methods to ensure robustness.
    """)
    
    # Feature importance method selection
    importance_method = st.selectbox(
        "Select Importance Method",
        options=['builtin', 'permutation', 'shap'],
        help="Different methods provide different perspectives on feature importance"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("Calculate Feature Importance"):
            with st.spinner(f"Calculating {importance_method} importance..."):
                fig = explainer.plot_feature_importance(
                    method=importance_method, 
                    top_n=20
                )
                if fig:
                    st.pyplot(fig)
                else:
                    st.warning(f"{importance_method} importance not available")
    
    with col2:
        st.markdown("**Method Descriptions:**")
        st.markdown("""
        - **Builtin**: Uses model's internal importance (fast)
        - **Permutation**: Measures performance drop when features are shuffled (robust)
        - **SHAP**: Game-theoretic approach providing consistent feature attributions (interpretable)
        
        **Key Insights:**
        - Different methods may rank features differently
        - Permutation importance is generally more reliable
        - SHAP values provide both global and local explanations
        """)

def _render_shap_analysis_tab(explainer):
    """Render SHAP analysis interface."""
    st.markdown("### SHAP (SHapley Additive exPlanations) Analysis")
    st.markdown("""
    SHAP provides a unified framework for interpreting model predictions by computing
    the contribution of each feature to individual predictions.
    """)
    
    analysis_type = st.selectbox(
        "Select SHAP Analysis Type",
        options=['summary_plot', 'waterfall', 'force_plot', 'dependence'],
        help="Different SHAP visualizations provide different insights"
    )
    
    if st.button("Generate SHAP Analysis"):
        with st.spinner("Generating SHAP analysis..."):
            try:
                if analysis_type == 'summary_plot':
                    fig = explainer.plot_shap_summary()
                elif analysis_type == 'waterfall':
                    instance_idx = st.number_input("Instance Index", 0, len(explainer.X_test)-1, 0)
                    fig = explainer.plot_shap_waterfall(instance_idx)
                elif analysis_type == 'force_plot':
                    instance_idx = st.number_input("Instance Index", 0, len(explainer.X_test)-1, 0)
                    fig = explainer.plot_shap_force(instance_idx)
                elif analysis_type == 'dependence':
                    feature_idx = st.number_input("Feature Index", 0, len(explainer.feature_names)-1, 0)
                    fig = explainer.plot_shap_dependence(feature_idx)
                
                if fig:
                    st.pyplot(fig)
                else:
                    st.warning("SHAP analysis not available for this model type")
            except Exception as e:
                st.error(f"Error generating SHAP analysis: {str(e)}")

def _render_instance_explanations_tab(explainer):
    """Render instance-level explanations."""
    st.markdown("### Instance-Level Explanations")
    st.markdown("""
    Analyze specific predictions to understand why the model classified
    a particular address as fraudulent or legitimate.
    """)
    
    # Instance selection
    instance_idx = st.number_input(
        "Select Instance to Explain",
        min_value=0,
        max_value=len(explainer.X_test) - 1,
        value=0,
        help="Choose which test instance to explain"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Explain Instance"):
            with st.spinner("Generating explanation..."):
                try:
                    # Get prediction
                    prediction = explainer.model.predict([explainer.X_test[instance_idx]])[0]
                    if hasattr(explainer.model, 'predict_proba'):
                        probability = explainer.model.predict_proba([explainer.X_test[instance_idx]])[0]
                        prob_text = f"Probability: {probability[1]:.3f}"
                    else:
                        prob_text = ""
                    
                    st.subheader("Prediction Result")
                    st.write(f"Predicted Class: {'Fraudulent' if prediction == 1 else 'Legitimate'}")
                    if prob_text:
                        st.write(prob_text)
                    st.write(f"Actual Class: {'Fraudulent' if explainer.y_test[instance_idx] == 1 else 'Legitimate'}")
                    
                    # Feature values for this instance
                    st.subheader("Feature Values")
                    feature_df = pd.DataFrame({
                        'Feature': explainer.feature_names,
                        'Value': explainer.X_test[instance_idx]
                    })
                    st.dataframe(feature_df)
                    
                except Exception as e:
                    st.error(f"Error explaining instance: {str(e)}")
    
    with col2:
        st.markdown("**Explanation Methods:**")
        st.markdown("""
        - **Feature Values**: Raw input values for the selected instance
        - **SHAP Values**: Contribution of each feature to the prediction
        - **LIME**: Local linear approximation around the instance
        - **Counterfactuals**: What changes would flip the prediction
        
        **Use Cases:**
        - Debugging model decisions
        - Building trust with stakeholders
        - Identifying bias or errors
        - Regulatory compliance
        """)

def _render_partial_dependence_tab(explainer):
    """Render partial dependence analysis."""
    st.markdown("### Partial Dependence Plots")
    st.markdown("""
    Partial dependence plots show how individual features affect predictions
    on average, while marginalizing over other features.
    """)
    
    # Feature selection for PDP
    feature_idx = st.selectbox(
        "Select Feature for Partial Dependence",
        options=list(range(len(explainer.feature_names))),
        format_func=lambda x: explainer.feature_names[x],
        help="Choose which feature to analyze"
    )
    
    if st.button("Generate Partial Dependence Plot"):
        with st.spinner("Calculating partial dependence..."):
            try:
                fig = explainer.plot_partial_dependence(feature_idx)
                if fig:
                    st.pyplot(fig)
                else:
                    st.warning("Partial dependence plot not available")
            except Exception as e:
                st.error(f"Error generating partial dependence plot: {str(e)}")
    
    st.markdown("""
    **Interpretation Guide:**
    - **Upward slope**: Higher feature values increase fraud probability
    - **Downward slope**: Higher feature values decrease fraud probability
    - **Flat line**: Feature has little impact on predictions
    - **Non-linear patterns**: Complex relationships between feature and target
    """)

def _render_model_analysis_tab(explainer):
    """Render comprehensive model analysis."""
    st.markdown("### Comprehensive Model Analysis")
    st.markdown("""
    This section provides a holistic view of model behavior and performance
    across different aspects of the fraud detection task.
    """)
    
    analysis_tabs = st.tabs([
        "Performance Metrics",
        "Feature Analysis",
        "Prediction Distribution",
        "Error Analysis"
    ])
    
    with analysis_tabs[0]:
        _render_performance_metrics(explainer)
    
    with analysis_tabs[1]:
        _render_feature_analysis(explainer)
    
    with analysis_tabs[2]:
        _render_prediction_distribution(explainer)
    
    with analysis_tabs[3]:
        _render_error_analysis(explainer)

def _render_performance_metrics(explainer):
    """Render detailed performance metrics."""
    st.markdown("#### Detailed Performance Metrics")
    
    try:
        y_pred = explainer.model.predict(explainer.X_test)
        
        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            accuracy = accuracy_score(explainer.y_test, y_pred)
            st.metric("Accuracy", f"{accuracy:.3f}")
        with col2:
            precision = precision_score(explainer.y_test, y_pred)
            st.metric("Precision", f"{precision:.3f}")
        with col3:
            recall = recall_score(explainer.y_test, y_pred)
            st.metric("Recall", f"{recall:.3f}")
        with col4:
            f1 = f1_score(explainer.y_test, y_pred)
            st.metric("F1-Score", f"{f1:.3f}")
        
        # Classification report
        st.markdown("#### Classification Report")
        report = classification_report(explainer.y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
        
    except Exception as e:
        st.error(f"Error calculating performance metrics: {str(e)}")

def _render_feature_analysis(explainer):
    """Render feature analysis."""
    st.markdown("#### Feature Analysis Summary")
    
    try:
        # Feature importance comparison
        st.markdown("**Feature Importance Comparison**")
        if st.button("Compare Importance Methods"):
            with st.spinner("Comparing feature importance methods..."):
                comparison_fig = compare_feature_importance_methods(
                    explainer.model, 
                    explainer.X_test, 
                    explainer.y_test,
                    explainer.feature_names
                )
                if comparison_fig:
                    st.pyplot(comparison_fig)
    
    except Exception as e:
        st.error(f"Error in feature analysis: {str(e)}")

def _render_prediction_distribution(explainer):
    """Render prediction distribution analysis."""
    st.markdown("#### Prediction Distribution")
    
    try:
        if hasattr(explainer.model, 'predict_proba'):
            y_proba = explainer.model.predict_proba(explainer.X_test)[:, 1]
            
            # Probability distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(y_proba[explainer.y_test == 0], alpha=0.5, label='Legitimate', bins=30)
            ax.hist(y_proba[explainer.y_test == 1], alpha=0.5, label='Fraudulent', bins=30)
            ax.set_xlabel('Fraud Probability')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Predicted Probabilities')
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("Probability distributions not available for this model type")
    
    except Exception as e:
        st.error(f"Error analyzing prediction distribution: {str(e)}")

def _render_error_analysis(explainer):
    """Render error analysis."""
    st.markdown("#### Error Analysis")
    
    try:
        y_pred = explainer.model.predict(explainer.X_test)
        
        # Confusion matrix-style analysis
        tp = ((explainer.y_test == 1) & (y_pred == 1)).sum()
        fp = ((explainer.y_test == 0) & (y_pred == 1)).sum()
        tn = ((explainer.y_test == 0) & (y_pred == 0)).sum()
        fn = ((explainer.y_test == 1) & (y_pred == 0)).sum()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**True Positives (Correctly identified fraud)**")
            st.metric("Count", tp)
            st.markdown("**False Positives (Incorrectly flagged as fraud)**")
            st.metric("Count", fp)
        
        with col2:
            st.markdown("**True Negatives (Correctly identified legitimate)**")
            st.metric("Count", tn)
            st.markdown("**False Negatives (Missed fraud)**")
            st.metric("Count", fn)
        
        # Error rate by class
        if fp + tn > 0:
            false_positive_rate = fp / (fp + tn)
            st.write(f"False Positive Rate: {false_positive_rate:.3f}")
        
        if fn + tp > 0:
            false_negative_rate = fn / (fn + tp)
            st.write(f"False Negative Rate: {false_negative_rate:.3f}")
    
    except Exception as e:
        st.error(f"Error in error analysis: {str(e)}")