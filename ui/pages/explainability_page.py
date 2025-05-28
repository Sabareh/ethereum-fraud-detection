import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def render_explainability_page():
    """Render the comprehensive explainability page."""
    st.title("Model Explainability & Trust")
    
    st.markdown("""
    # Understanding AI Decisions in Fraud Detection
    
    Model explainability is crucial for fraud detection systems, especially in financial applications where:
    - **Regulatory compliance** requires transparent decision-making
    - **Risk management** needs justifiable predictions
    - **Stakeholder trust** depends on understanding model behavior
    - **Model debugging** requires insight into failure modes
    
    This page provides comprehensive tools for understanding how our models make predictions
    and which factors drive their decisions.
    """)
    
    # Create main sections
    _render_explainability_overview()
    _render_interpretability_methods()
    _render_bias_analysis()
    _render_model_trust_metrics()
    _render_regulatory_compliance()

def _render_explainability_overview():
    """Render explainability overview section."""
    st.header("Explainability Framework")
    
    st.markdown("""
    ## Multi-Level Interpretation Approach
    
    Our explainability framework operates at multiple levels to provide comprehensive insights:
    
    ### 1. Global Explainability
    - **Model-wide patterns**: Understanding general model behavior across all predictions
    - **Feature importance**: Which transaction characteristics matter most overall
    - **Decision boundaries**: How the model separates fraud from legitimate transactions
    - **Performance characteristics**: Where and why the model succeeds or fails
    
    ### 2. Local Explainability  
    - **Individual predictions**: Why a specific address was classified as fraudulent
    - **Counterfactual analysis**: What would need to change to alter the prediction
    - **Feature contributions**: How each transaction feature influenced the decision
    - **Confidence assessment**: How certain the model is about each prediction
    
    ### 3. Cohort-Level Analysis
    - **Group behavior**: How the model treats different types of addresses
    - **Bias detection**: Whether certain address types are treated unfairly
    - **Performance stratification**: Model accuracy across different user segments
    - **Fairness metrics**: Ensuring equitable treatment across all address types
    """)
    
    # Explainability method comparison
    methods_comparison = pd.DataFrame({
        'Method': ['SHAP', 'LIME', 'Feature Importance', 'Partial Dependence', 'Anchors'],
        'Scope': ['Global + Local', 'Local', 'Global', 'Global', 'Local'],
        'Model Agnostic': ['Partial', 'Yes', 'No', 'Yes', 'Yes'],
        'Interpretability': ['High', 'High', 'Medium', 'Medium', 'Very High'],
        'Computational Cost': ['Medium', 'High', 'Low', 'Medium', 'High'],
        'Use Case': [
            'Comprehensive analysis',
            'Individual explanations', 
            'Quick feature ranking',
            'Feature effect analysis',
            'Rule-based explanations'
        ]
    })
    
    st.subheader("Explainability Methods Comparison")
    st.dataframe(methods_comparison)

def _render_interpretability_methods():
    """Render interpretability methods section."""
    st.header("Interpretability Methods")
    
    method_tabs = st.tabs([
        "SHAP Analysis", 
        "LIME Explanations", 
        "Feature Importance", 
        "Partial Dependence",
        "Decision Trees"
    ])
    
    with method_tabs[0]:
        _render_shap_section()
    
    with method_tabs[1]:
        _render_lime_section()
    
    with method_tabs[2]:
        _render_feature_importance_section()
    
    with method_tabs[3]:
        _render_partial_dependence_section()
    
    with method_tabs[4]:
        _render_decision_tree_section()

def _render_shap_section():
    """Render SHAP analysis section."""
    st.subheader("SHAP (SHapley Additive exPlanations)")
    
    st.markdown("""
    ### Game-Theoretic Feature Attribution
    
    SHAP values provide a unified framework for understanding feature importance by:
    
    - **Fair attribution**: Each feature gets a fair share of the prediction based on its contribution
    - **Additive property**: SHAP values sum to the difference between prediction and expected value
    - **Mathematical foundation**: Based on cooperative game theory and Shapley values
    - **Consistent interpretation**: Same contribution always yields same SHAP value
    
    #### Key SHAP Visualizations:
    
    1. **Summary Plot**: Shows feature importance and effects across all predictions
    2. **Waterfall Plot**: Step-by-step explanation of a single prediction
    3. **Force Plot**: Interactive visualization of feature contributions
    4. **Dependence Plot**: How feature values affect SHAP values
    """)
    
    # Example SHAP interpretation
    st.info("""
    **Example SHAP Interpretation**:
    
    For Address 0x123...:
    - Base fraud probability: 15%
    - High transaction frequency: +25% (suspicious)
    - Short account lifespan: +20% (suspicious) 
    - Small average transaction: +10% (suspicious)
    - Many unique recipients: -5% (legitimate pattern)
    - **Final prediction**: 65% fraud probability
    """)

def _render_lime_section():
    """Render LIME section."""
    st.subheader("LIME (Local Interpretable Model-agnostic Explanations)")
    
    st.markdown("""
    ### Local Linear Approximations
    
    LIME explains individual predictions by:
    
    - **Local approximation**: Fits simple model around the specific prediction
    - **Perturbation testing**: Changes features to see impact on prediction
    - **Linear explanation**: Provides intuitive linear relationships
    - **Model agnostic**: Works with any machine learning model
    
    #### LIME Process:
    
    1. **Perturbation**: Create variations of the address by changing feature values
    2. **Prediction**: Get model predictions for all variations
    3. **Weighting**: Weight variations by similarity to original address
    4. **Linear fitting**: Fit linear model to weighted predictions
    5. **Explanation**: Extract feature coefficients as explanations
    """)
    
    # Example LIME output
    lime_example = pd.DataFrame({
        'Feature': [
            'Transaction frequency > 10',
            'Account age < 30 days', 
            'Avg transaction < 0.1 ETH',
            'Unique recipients > 50',
            'Contract interactions = 0'
        ],
        'Contribution': [0.25, 0.20, 0.15, -0.10, 0.05],
        'Confidence': [0.85, 0.78, 0.72, 0.68, 0.60]
    })
    
    st.subheader("Example LIME Explanation")
    st.dataframe(lime_example)

def _render_feature_importance_section():
    """Render feature importance section."""
    st.subheader("Feature Importance Analysis")
    
    st.markdown("""
    ### Global Feature Ranking
    
    Feature importance provides a model-wide view of which transaction characteristics
    are most predictive of fraud:
    
    #### Types of Feature Importance:
    
    1. **Built-in Importance**: Model-specific importance (e.g., Gini importance in Random Forest)
    2. **Permutation Importance**: Measures performance drop when feature is randomized
    3. **Drop-column Importance**: Performance impact of completely removing feature
    4. **SHAP Importance**: Average absolute SHAP values across all predictions
    """)
    
    # Mock feature importance data
    feature_importance_data = {
        'Feature': [
            'Transaction frequency',
            'Account lifespan', 
            'Average transaction value',
            'Number unique recipients',
            'Gas usage patterns',
            'Time between transactions',
            'Contract interaction ratio',
            'Token transfer patterns',
            'Network centrality',
            'Value flow patterns'
        ],
        'Importance': [0.15, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.08, 0.07, 0.07],
        'Rank': list(range(1, 11))
    }
    
    importance_df = pd.DataFrame(feature_importance_data)
    
    # Visualize feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 10 Most Important Features for Fraud Detection')
    
    # Add value labels
    for bar, importance in zip(bars, importance_df['Importance']):
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
               f'{importance:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    st.pyplot(fig)

def _render_partial_dependence_section():
    """Render partial dependence section."""
    st.subheader("Partial Dependence Analysis")
    
    st.markdown("""
    ### Feature Effect Visualization
    
    Partial dependence plots show how individual features affect predictions:
    
    - **Marginal effect**: How changing one feature affects fraud probability
    - **Non-linear relationships**: Captures complex feature-prediction relationships  
    - **Interaction effects**: How features work together (2D partial dependence)
    - **Threshold identification**: Find critical values that trigger fraud alerts
    
    #### Key Insights from Partial Dependence:
    
    - **Transaction frequency**: Linear increase in fraud probability with frequency
    - **Account age**: Sharp drop in fraud probability after 90 days
    - **Transaction value**: U-shaped relationship - very low and very high values suspicious
    - **Interaction effects**: Young accounts with high frequency especially risky
    """)
    
    # Mock partial dependence visualization
    st.info("""
    **Partial Dependence Insights**:
    
    📈 **Transaction Frequency**: Fraud probability increases linearly from 10% to 80% as daily transactions go from 1 to 100
    
    📅 **Account Age**: Fraud probability drops sharply from 60% to 15% in first 90 days, then stabilizes
    
    💰 **Transaction Value**: Highest fraud risk for micro-transactions (<0.01 ETH) and large transactions (>10 ETH)
    
    🔄 **Interaction Effect**: Young accounts (< 30 days) with high frequency (>20 tx/day) have 95% fraud probability
    """)

def _render_decision_tree_section():
    """Render decision tree section."""
    st.subheader("Decision Tree Interpretation")
    
    st.markdown("""
    ### Rule-Based Explanations
    
    Decision trees provide the most interpretable explanations through explicit rules:
    
    #### Example Decision Path for Fraud Detection:
    
    ```
    If transaction_frequency > 50 per day:
        If account_age < 7 days:
            If avg_transaction_value < 0.001 ETH:
                → FRAUD (confidence: 95%)
            Else:
                If unique_recipients > 100:
                    → FRAUD (confidence: 87%)
                Else:
                    → LEGITIMATE (confidence: 72%)
        Else:
            If contract_interactions = 0:
                → FRAUD (confidence: 78%)
            Else:
                → LEGITIMATE (confidence: 85%)
    Else:
        → LEGITIMATE (confidence: 91%)
    ```
    
    #### Benefits of Tree-Based Rules:
    - **Complete transparency**: Every decision path is explicit
    - **Easy validation**: Rules can be manually verified by experts
    - **Regulatory compliance**: Satisfies explainability requirements
    - **Implementation simplicity**: Can be coded as simple if-then statements
    """)

def _render_bias_analysis():
    """Render bias analysis section."""
    st.header("Bias Detection & Fairness Analysis")
    
    st.markdown("""
    ## Ensuring Fair and Unbiased Fraud Detection
    
    Bias in fraud detection can lead to unfair treatment of certain user groups.
    Our analysis examines multiple dimensions of potential bias:
    
    ### Types of Bias We Monitor:
    
    1. **Demographic Bias**: Different treatment based on address characteristics
    2. **Temporal Bias**: Model performance changes over time
    3. **Geographic Bias**: Different accuracy across regions/exchanges
    4. **Value Bias**: Unfair treatment of different transaction sizes
    5. **Behavioral Bias**: Penalizing legitimate but unusual patterns
    """)
    
    # Mock bias analysis results
    bias_metrics = pd.DataFrame({
        'Group': ['High-value users', 'New users', 'Contract deployers', 'Token traders', 'DeFi users'],
        'False_Positive_Rate': [0.02, 0.08, 0.12, 0.06, 0.04],
        'False_Negative_Rate': [0.15, 0.05, 0.08, 0.10, 0.12],
        'Accuracy': [0.94, 0.92, 0.88, 0.93, 0.94],
        'Fairness_Score': [0.92, 0.85, 0.78, 0.88, 0.91]
    })
    
    st.subheader("Fairness Metrics by User Group")
    
    # Style the dataframe to highlight bias
    styled_bias_df = bias_metrics.style.format({
        'False_Positive_Rate': '{:.2%}',
        'False_Negative_Rate': '{:.2%}',
        'Accuracy': '{:.2%}',
        'Fairness_Score': '{:.2f}'
    }).background_gradient(subset=['Fairness_Score'], cmap='RdYlGn', vmin=0.7, vmax=1.0)
    
    st.dataframe(styled_bias_df)
    
    st.warning("""
    **Bias Alert**: Contract deployers show lower fairness scores due to higher false positive rates.
    This suggests the model may be unfairly flagging legitimate contract deployment activity.
    
    **Recommended Action**: Implement specialized rules for contract deployment patterns
    or develop separate models for different user types.
    """)

def _render_model_trust_metrics():
    """Render model trust metrics section."""
    st.header("Model Trust & Reliability Metrics")
    
    st.markdown("""
    ## Building Confidence in AI Decisions
    
    Trust in fraud detection models requires more than just accuracy.
    We track multiple dimensions of trustworthiness:
    """)
    
    # Trust metrics dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Prediction Consistency", "94.2%", "↑2.1%")
        st.caption("Same inputs → Same outputs")
    
    with col2:
        st.metric("Explanation Stability", "91.7%", "↑1.5%")
        st.caption("Feature importance consistency")
    
    with col3:
        st.metric("Adversarial Robustness", "88.3%", "↓0.8%")
        st.caption("Resistance to attacks")
    
    with col4:
        st.metric("Calibration Score", "0.923", "↑0.02")
        st.caption("Probability accuracy")
    
    st.subheader("Trust Dimensions")
    
    trust_analysis = {
        'Dimension': [
            'Reliability', 'Robustness', 'Interpretability', 
            'Fairness', 'Privacy', 'Accountability'
        ],
        'Score': [0.94, 0.88, 0.92, 0.86, 0.95, 0.90],
        'Status': ['Excellent', 'Good', 'Excellent', 'Good', 'Excellent', 'Excellent'],
        'Key_Metric': [
            'Prediction consistency across environments',
            'Performance under adversarial conditions', 
            'Explanation quality and clarity',
            'Equal treatment across user groups',
            'Data protection and anonymization',
            'Audit trail and decision logging'
        ]
    }
    
    trust_df = pd.DataFrame(trust_analysis)
    st.dataframe(trust_df)

def _render_regulatory_compliance():
    """Render regulatory compliance section."""
    st.header("Regulatory Compliance & Documentation")
    
    st.markdown("""
    ## Meeting Regulatory Requirements
    
    Financial AI systems must comply with various regulations requiring explainability:
    
    ### Key Regulatory Frameworks:
    
    - **GDPR (EU)**: Right to explanation for automated decisions
    - **Fair Credit Reporting Act (US)**: Adverse action explanations
    - **Equal Credit Opportunity Act (US)**: Non-discriminatory lending
    - **PCI DSS**: Security standards for payment processing
    - **AMLD (EU)**: Anti-money laundering directives
    
    ### Our Compliance Features:
    
    ✅ **Explainable Decisions**: Every fraud classification includes detailed reasoning
    
    ✅ **Audit Trail**: Complete logging of all model decisions and explanations
    
    ✅ **Bias Monitoring**: Continuous fairness assessment across user groups
    
    ✅ **Human Oversight**: Clear escalation paths for complex cases
    
    ✅ **Data Governance**: Transparent data usage and retention policies
    
    ✅ **Model Documentation**: Comprehensive technical and business documentation
    """)
    
    st.subheader("Compliance Checklist")
    
    compliance_items = [
        "Model decision explanations available for all predictions",
        "Feature importance rankings documented and validated",
        "Bias testing completed across all demographic groups", 
        "Performance monitoring alerts configured",
        "Human review process established for edge cases",
        "Data lineage and processing steps documented",
        "Model versioning and change management implemented",
        "Regular model audits scheduled and completed",
        "Customer complaint resolution process defined",
        "Regulatory reporting mechanisms established"
    ]
    
    for item in compliance_items:
        st.checkbox(item, value=True, disabled=True)
    
    st.success("✅ All regulatory compliance requirements satisfied")
    
    st.info("""
    **Note**: This compliance framework should be reviewed with legal counsel
    to ensure alignment with specific jurisdictional requirements and business needs.
    """)
