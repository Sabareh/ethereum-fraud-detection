import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve, train_test_split
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

from model_utils import get_supervised_models, get_unsupervised_models, load_data

def render_learning_curves_tab():
    """Render learning curves analysis tab."""
    st.header("📈 Learning Curves")
    
    st.markdown("""
    ### Model Learning & Performance Analysis
    
    Analyze how models learn and perform across different conditions including
    validation curves, sample size impact, and feature learning patterns for both supervised and unsupervised models.
    """)
    
    # Learning curves subtabs
    learning_tabs = st.tabs([
        "Traditional Learning Curves",
        "Validation Curves", 
        "Sample Size Impact",
        "Feature Learning Analysis"
    ])
    
    with learning_tabs[0]:
        _render_traditional_learning_curves()
    
    with learning_tabs[1]:
        _render_validation_curves()
    
    with learning_tabs[2]:
        _render_sample_size_impact()
    
    with learning_tabs[3]:
        _render_feature_learning_analysis()

def _render_traditional_learning_curves():
    """Render traditional learning curves analysis."""
    st.subheader("Traditional Learning Curves")
    
    st.markdown("""
    ### Training vs Validation Performance
    
    Traditional learning curves show how model performance changes as training set size increases.
    They help identify whether models would benefit from more data and detect overfitting/underfitting.
    """)
    
    # Model type selection
    model_type = st.selectbox(
        "Select model type for learning curves",
        options=["Supervised", "Unsupervised"],
        key="tlc_model_type"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if model_type == "Supervised":
            supervised_models = get_supervised_models()
            selected_models = st.multiselect(
                "Select supervised models for learning curves",
                options=list(supervised_models.keys()),
                default=list(supervised_models.keys())[:2],
                key="tlc_supervised_models"
            )
            
            if st.button("Generate Supervised Learning Curves", key="tlc_supervised_btn"):
                if selected_models:
                    X, y = load_data(use_synthetic=True)
                    
                    fig, axes = plt.subplots(len(selected_models), 1, figsize=(12, 6 * len(selected_models)))
                    if len(selected_models) == 1:
                        axes = [axes]
                    
                    for idx, model_name in enumerate(selected_models):
                        model = supervised_models[model_name]
                        
                        # Generate learning curve
                        train_sizes, train_scores, val_scores = learning_curve(
                            model, X, y, cv=5, n_jobs=-1, 
                            train_sizes=np.linspace(0.1, 1.0, 10),
                            scoring='f1'
                        )
                        
                        train_mean = np.mean(train_scores, axis=1)
                        train_std = np.std(train_scores, axis=1)
                        val_mean = np.mean(val_scores, axis=1)
                        val_std = np.std(val_scores, axis=1)
                        
                        ax = axes[idx]
                        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score', linewidth=2)
                        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
                        
                        ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score', linewidth=2)
                        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
                        
                        ax.set_xlabel('Training Set Size')
                        ax.set_ylabel('F1 Score')
                        ax.set_title(f'Learning Curve: {model_name}')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
        
        else:  # Unsupervised
            unsupervised_models = get_unsupervised_models()
            selected_models = st.multiselect(
                "Select unsupervised models for learning curves",
                options=list(unsupervised_models.keys()),
                default=list(unsupervised_models.keys())[:2],
                key="tlc_unsupervised_models"
            )
            
            metric_choice = st.selectbox(
                "Select evaluation metric",
                options=["Silhouette Score", "Inertia (if available)", "Fraud Detection Rate"],
                key="tlc_unsup_metric"
            )
            
            if st.button("Generate Unsupervised Learning Curves", key="tlc_unsupervised_btn"):
                if selected_models:
                    X, y = load_data(use_synthetic=True)
                    
                    # Preprocess data
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    fig, axes = plt.subplots(len(selected_models), 1, figsize=(12, 6 * len(selected_models)))
                    if len(selected_models) == 1:
                        axes = [axes]
                    
                    train_sizes = np.linspace(0.1, 1.0, 10)
                    
                    for idx, model_name in enumerate(selected_models):
                        model = unsupervised_models[model_name]
                        
                        scores = []
                        actual_sizes = []
                        
                        for size_fraction in train_sizes:
                            n_samples = int(size_fraction * len(X_scaled))
                            X_subset = X_scaled[:n_samples]
                            y_subset = y[:n_samples]
                            
                            try:
                                model_copy = clone(model)
                                labels = model_copy.fit_predict(X_subset)
                                
                                if metric_choice == "Silhouette Score":
                                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                                    if n_clusters > 1:
                                        score = silhouette_score(X_subset, labels)
                                    else:
                                        score = 0
                                elif metric_choice == "Inertia (if available)":
                                    if hasattr(model_copy, 'inertia_'):
                                        score = -model_copy.inertia_  # Negative because lower is better
                                    else:
                                        score = 0
                                else:  # Fraud Detection Rate
                                    unique_labels = set(labels)
                                    max_fraud_rate = 0
                                    for cluster_id in unique_labels:
                                        if cluster_id != -1:
                                            mask = labels == cluster_id
                                            if mask.sum() > 0:
                                                fraud_rate = y_subset[mask].mean()
                                                max_fraud_rate = max(max_fraud_rate, fraud_rate)
                                    score = max_fraud_rate
                                
                                scores.append(score)
                                actual_sizes.append(n_samples)
                                
                            except Exception as e:
                                scores.append(0)
                                actual_sizes.append(n_samples)
                        
                        ax = axes[idx]
                        ax.plot(actual_sizes, scores, 'o-', color='green', linewidth=2, markersize=6)
                        ax.set_xlabel('Training Set Size')
                        ax.set_ylabel(metric_choice)
                        ax.set_title(f'Unsupervised Learning Curve: {model_name}')
                        ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
    
    with col2:
        st.markdown("### Learning Curve Interpretation")
        st.info("""
        **Supervised Models:**
        
        **High Training, Low Validation:**
        - Model is overfitting
        - More data or regularization needed
        
        **Both Curves Low:**
        - Model is underfitting
        - Need more complex model or features
        
        **Converging Curves:**
        - Good generalization
        - Model is well-tuned
        
        **Unsupervised Models:**
        
        **Increasing Metric:**
        - Model benefits from more data
        - Quality improves with size
        
        **Plateauing Metric:**
        - Optimal data size reached
        - Diminishing returns from more data
        """)

def _render_validation_curves():
    """Render validation curves analysis."""
    st.subheader("Validation Curves")
    
    st.markdown("""
    ### Hyperparameter Impact Analysis
    
    Validation curves show how model performance changes with hyperparameter values.
    They help identify optimal hyperparameter settings and detect overfitting for both supervised and unsupervised models.
    """)
    
    # Model type selection
    model_type = st.selectbox(
        "Select model type for validation curve",
        options=["Supervised", "Unsupervised"],
        key="vc_model_type"
    )
    
    if model_type == "Supervised":
        supervised_models = get_supervised_models()
        model_name = st.selectbox("Select supervised model for validation curve",
                                  options=list(supervised_models.keys()), key="vc_model")
        selected_model = supervised_models[model_name]
        
        params = selected_model.get_params()
        numeric_params = [k for k, v in params.items() if isinstance(v, (int, float)) and v is not None]
        
        if not numeric_params:
            st.warning("Selected model has no numeric hyperparameters to analyze.")
            return
        
        param_name = st.selectbox("Choose hyperparameter", options=numeric_params[:5], key="vc_param")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if param_name:
                current_value = params[param_name]
                
                # Smart range selection based on parameter type
                if 'alpha' in param_name.lower() or 'learning_rate' in param_name.lower():
                    # For regularization parameters, use log scale
                    low = max(0.001, current_value / 100)
                    high = current_value * 100
                    param_range = np.logspace(np.log10(low), np.log10(high), 10)
                elif 'n_estimators' in param_name.lower():
                    # For number of estimators
                    low = max(10, current_value // 2)
                    high = current_value * 3
                    param_range = np.linspace(low, high, 8, dtype=int)
                else:
                    # General case
                    low = max(0.1, current_value * 0.1) if current_value > 0 else 0.1
                    high = current_value * 3 if current_value > 0 else 10
                    param_range = np.linspace(low, high, 8)
                
                st.write(f"Testing {param_name} in range: {param_range[0]:.3f} to {param_range[-1]:.3f}")
                
                if st.button("Plot Supervised Validation Curve", key="vc_btn"):
                    with st.spinner("Generating validation curves..."):
                        X, y = load_data(use_synthetic=True)
                        
                        try:
                            train_scores, val_scores = validation_curve(
                                clone(selected_model),
                                X, y,
                                param_name=param_name,
                                param_range=param_range,
                                cv=5,
                                scoring="f1",
                                n_jobs=-1
                            )
                            
                            train_mean = np.mean(train_scores, axis=1)
                            train_std = np.std(train_scores, axis=1)
                            val_mean = np.mean(val_scores, axis=1)
                            val_std = np.std(val_scores, axis=1)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Plot validation curves
                            ax.plot(param_range, train_mean, 'o-', color='blue', label='Training F1', linewidth=2)
                            ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
                            
                            ax.plot(param_range, val_mean, 'o-', color='red', label='Validation F1', linewidth=2)
                            ax.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
                            
                            # Mark current value
                            ax.axvline(x=current_value, color='green', linestyle='--', alpha=0.7, label=f'Current: {current_value}')
                            
                            # Mark optimal value
                            optimal_idx = np.argmax(val_mean)
                            optimal_value = param_range[optimal_idx]
                            ax.axvline(x=optimal_value, color='orange', linestyle='--', alpha=0.7, label=f'Optimal: {optimal_value:.3f}')
                            
                            ax.set_xlabel(param_name)
                            ax.set_ylabel("F1 Score")
                            ax.set_title(f"Validation Curve – {model_name}")
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            
                            # Use log scale for certain parameters
                            if 'alpha' in param_name.lower() or 'learning_rate' in param_name.lower():
                                ax.set_xscale('log')
                            
                            st.pyplot(fig)
                            
                            # Provide recommendations
                            st.subheader("Optimization Recommendations")
                            improvement = (val_mean[optimal_idx] - val_mean[np.where(param_range == current_value)[0][0] if current_value in param_range else 0]) if current_value in param_range else 0
                            
                            if improvement > 0.01:
                                st.success(f"✅ Optimal {param_name}: {optimal_value:.4f} (improvement: +{improvement:.4f})")
                            elif improvement > 0.005:
                                st.info(f"📊 Slight improvement possible with {param_name}: {optimal_value:.4f}")
                            else:
                                st.info(f"Current {param_name} value is near optimal")
                                
                        except Exception as e:
                            st.error(f"Error generating validation curve: {str(e)}")
        
        with col2:
            st.markdown("### Validation Curve Interpretation")
            st.info("""
            **Key Patterns:**
            
            **Underfitting:**
            - Both curves low and similar
            - Need to decrease regularization
            
            **Overfitting:**
            - Large gap between curves
            - Need to increase regularization
            
            **Optimal Point:**
            - Maximum validation score
            - Minimal gap between curves
            """)
    
    else:  # Unsupervised
        unsupervised_models = get_unsupervised_models()
        model_name = st.selectbox("Select unsupervised model for validation curve",
                                  options=list(unsupervised_models.keys()), key="vc_unsup_model")
        selected_model = unsupervised_models[model_name]
        
        params = selected_model.get_params()
        numeric_params = [k for k, v in params.items() if isinstance(v, (int, float)) and v is not None]
        
        if not numeric_params:
            st.warning("Selected model has no numeric hyperparameters to analyze.")
            return
        
        param_name = st.selectbox("Choose hyperparameter", options=numeric_params[:5], key="vc_unsup_param")
        
        metric_type = st.selectbox(
            "Select evaluation metric",
            options=["Silhouette Score", "Fraud Detection Effectiveness"],
            key="vc_unsup_metric"
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if param_name:
                current_value = params[param_name]
                
                # Smart range selection
                if 'n_clusters' in param_name.lower():
                    low = max(2, current_value - 3)
                    high = current_value + 5
                    param_range = np.arange(low, high + 1, dtype=int)
                elif 'eps' in param_name.lower():
                    low = max(0.1, current_value * 0.5)
                    high = current_value * 2
                    param_range = np.linspace(low, high, 8)
                else:
                    low = max(0.1, current_value * 0.5) if current_value > 0 else 0.1
                    high = current_value * 2 if current_value > 0 else 10
                    param_range = np.linspace(low, high, 8)
                
                st.write(f"Testing {param_name} in range: {param_range[0]:.3f} to {param_range[-1]:.3f}")
                
                if st.button("Plot Unsupervised Validation Curve", key="vc_unsup_btn"):
                    with st.spinner("Generating unsupervised validation curves..."):
                        X, y = load_data(use_synthetic=True)
                        
                        # Preprocess data
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        scores = []
                        
                        for param_value in param_range:
                            try:
                                # Create model with modified parameter
                                model_params = params.copy()
                                model_params[param_name] = param_value
                                model = type(selected_model)(**model_params)
                                
                                labels = model.fit_predict(X_scaled)
                                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                                
                                if metric_type == "Silhouette Score" and n_clusters > 1:
                                    score = silhouette_score(X_scaled, labels)
                                elif metric_type == "Fraud Detection Effectiveness":
                                    # Calculate fraud detection effectiveness
                                    unique_labels = set(labels)
                                    cluster_fraud_rates = []
                                    for cluster_id in unique_labels:
                                        if cluster_id != -1:
                                            mask = labels == cluster_id
                                            if mask.sum() > 0:
                                                fraud_rate = y[mask].mean()
                                                cluster_fraud_rates.append(fraud_rate)
                                    
                                    if cluster_fraud_rates:
                                        score = max(cluster_fraud_rates)
                                    else:
                                        score = 0
                                else:
                                    score = 0
                                
                                scores.append(score)
                                
                            except Exception as e:
                                scores.append(0)
                        
                        # Plot results
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        ax.plot(param_range, scores, 'o-', color='blue', linewidth=2, markersize=6)
                        ax.axvline(x=current_value, color='green', linestyle='--', alpha=0.7, label=f'Current: {current_value}')
                        
                        # Mark optimal value
                        if scores:
                            optimal_idx = np.argmax(scores)
                            optimal_value = param_range[optimal_idx]
                            ax.axvline(x=optimal_value, color='orange', linestyle='--', alpha=0.7, label=f'Optimal: {optimal_value:.3f}')
                        
                        ax.set_xlabel(param_name)
                        ax.set_ylabel(metric_type)
                        ax.set_title(f"Unsupervised Validation Curve – {model_name}")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                        
                        # Provide recommendations
                        if scores:
                            optimal_score = max(scores)
                            current_score_idx = np.argmin(np.abs(param_range - current_value))
                            current_score = scores[current_score_idx]
                            improvement = optimal_score - current_score
                            
                            st.subheader("Optimization Recommendations")
                            if improvement > 0.05:
                                st.success(f"✅ Optimal {param_name}: {optimal_value:.4f} (improvement: +{improvement:.4f})")
                            elif improvement > 0.01:
                                st.info(f"📊 Slight improvement possible with {param_name}: {optimal_value:.4f}")
                            else:
                                st.info(f"Current {param_name} value is near optimal")
        
        with col2:
            st.markdown("### Unsupervised Validation Guide")
            st.info("""
            **Interpretation Guidelines:**
            
            **Silhouette Score:**
            - Higher values indicate better clustering
            - Look for peak values
            - Avoid overly complex models
            
            **Fraud Detection:**
            - Higher max fraud rate is better
            - Shows ability to isolate fraud patterns
            
            **Parameter Selection:**
            - Consider both metrics
            - Validate on multiple datasets
            - Monitor cluster stability
            """)

def _render_sample_size_impact():
    """Render sample size impact analysis."""
    st.subheader("Sample Size Impact")
    
    st.markdown("""
    ### Data Requirements Analysis
    
    Understand how much training data is needed for optimal performance
    and whether collecting more data would be beneficial for both supervised and unsupervised models.
    """)
    
    # Model type selection
    model_type = st.selectbox(
        "Select model type for sample size analysis",
        options=["Supervised", "Unsupervised"],
        key="ssi_model_type"
    )
    
    if model_type == "Supervised":
        supervised_models = get_supervised_models()
        model_name = st.selectbox("Select supervised model", options=list(supervised_models.keys()), key="ssi_model")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fractions = np.linspace(0.1, 0.9, 9)  # Fixed to avoid train_size=1.0
            
            if st.button("Plot Supervised Sample Size Impact", key="ssi_btn"):
                with st.spinner("Analyzing sample size impact..."):
                    X, y = load_data(use_synthetic=True)
                    train_scores = []
                    test_scores = []
                    data_sizes = []
                    
                    for f in fractions:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, train_size=f, random_state=42, stratify=y
                        )
                        
                        model = clone(supervised_models[model_name])
                        model.fit(X_train, y_train)
                        
                        train_score = model.score(X_train, y_train)
                        test_score = model.score(X_test, y_test)
                        
                        train_scores.append(train_score)
                        test_scores.append(test_score)
                        data_sizes.append(len(X_train))
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Plot 1: Performance vs percentage
                    ax1.plot(fractions * 100, train_scores, 'o-', label='Training Accuracy', color='blue', linewidth=2)
                    ax1.plot(fractions * 100, test_scores, 'o-', label='Test Accuracy', color='red', linewidth=2)
                    ax1.set_xlabel("Training Set Size (%)")
                    ax1.set_ylabel("Accuracy")
                    ax1.set_title(f"Performance vs Data Size – {model_name}")
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Plot 2: Performance vs absolute numbers
                    ax2.plot(data_sizes, train_scores, 'o-', label='Training Accuracy', color='blue', linewidth=2)
                    ax2.plot(data_sizes, test_scores, 'o-', label='Test Accuracy', color='red', linewidth=2)
                    ax2.set_xlabel("Number of Training Samples")
                    ax2.set_ylabel("Accuracy")
                    ax2.set_title(f"Performance vs Sample Count – {model_name}")
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Analysis
                    performance_improvement = test_scores[-1] - test_scores[0]
                    diminishing_returns = test_scores[-1] - test_scores[-3] < 0.01
                    
                    st.subheader("Data Collection Recommendations")
                    if performance_improvement > 0.05:
                        st.success(f"✅ Significant improvement with more data (+{performance_improvement:.3f})")
                        if diminishing_returns:
                            st.info("📊 Returns are diminishing - current data size may be sufficient")
                        else:
                            st.info("📈 Consider collecting more data for further improvements")
                    else:
                        st.info("Current data size appears sufficient for this model")
        
        with col2:
            st.markdown("### Sample Size Guidelines")
            st.info("""
            **Data Collection Strategy:**
            
            **Steep Curve:**
            - More data will help significantly
            - Prioritize data collection
            
            **Flat Curve:**
            - Model has reached capacity
            - Focus on feature engineering
            
            **Diminishing Returns:**
            - Current data size is adequate
            - Cost vs benefit analysis needed
            """)
    
    else:  # Unsupervised
        unsupervised_models = get_unsupervised_models()
        model_name = st.selectbox("Select unsupervised model", options=list(unsupervised_models.keys()), key="ssi_unsup_model")
        
        metric_type = st.selectbox(
            "Select evaluation metric",
            options=["Silhouette Score", "Fraud Detection Effectiveness", "Cluster Stability"],
            key="ssi_unsup_metric"
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fractions = np.linspace(0.1, 0.9, 9)
            
            if st.button("Plot Unsupervised Sample Size Impact", key="ssi_unsup_btn"):
                with st.spinner("Analyzing unsupervised sample size impact..."):
                    X, y = load_data(use_synthetic=True)
                    
                    # Preprocess data
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    scores = []
                    cluster_counts = []
                    data_sizes = []
                    stability_scores = []
                    
                    # Get baseline clustering for stability comparison
                    baseline_model = clone(unsupervised_models[model_name])
                    baseline_labels = baseline_model.fit_predict(X_scaled)
                    
                    for f in fractions:
                        # Get subset of data
                        n_samples = int(f * len(X_scaled))
                        X_subset = X_scaled[:n_samples]
                        y_subset = y[:n_samples]
                        
                        model = clone(unsupervised_models[model_name])
                        labels = model.fit_predict(X_subset)
                        
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        cluster_counts.append(n_clusters)
                        data_sizes.append(n_samples)
                        
                        if metric_type == "Silhouette Score" and n_clusters > 1:
                            score = silhouette_score(X_subset, labels)
                        elif metric_type == "Fraud Detection Effectiveness":
                            # Calculate fraud detection effectiveness
                            unique_labels = set(labels)
                            cluster_fraud_rates = []
                            for cluster_id in unique_labels:
                                if cluster_id != -1:
                                    mask = labels == cluster_id
                                    if mask.sum() > 0:
                                        fraud_rate = y_subset[mask].mean()
                                        cluster_fraud_rates.append(fraud_rate)
                            
                            if cluster_fraud_rates:
                                score = max(cluster_fraud_rates)
                            else:
                                score = 0
                        elif metric_type == "Cluster Stability":
                            # Compare with baseline clustering
                            overlap_size = min(len(labels), len(baseline_labels))
                            score = adjusted_rand_score(labels[:overlap_size], baseline_labels[:overlap_size])
                        else:
                            score = 0
                        
                        scores.append(score)
                        
                        # Calculate stability compared to previous size
                        if len(scores) > 1:
                            stability_scores.append(abs(scores[-1] - scores[-2]))
                    
                    # Create visualizations
                    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                    
                    # Plot 1: Main metric vs data size
                    axes[0,0].plot(data_sizes, scores, 'o-', color='blue', linewidth=2, markersize=6)
                    axes[0,0].set_xlabel("Training Set Size")
                    axes[0,0].set_ylabel(metric_type)
                    axes[0,0].set_title(f"{metric_type} vs Data Size")
                    axes[0,0].grid(True, alpha=0.3)
                    
                    # Plot 2: Number of clusters vs data size
                    axes[0,1].plot(data_sizes, cluster_counts, 'o-', color='green', linewidth=2, markersize=6)
                    axes[0,1].set_xlabel("Training Set Size")
                    axes[0,1].set_ylabel("Number of Clusters")
                    axes[0,1].set_title("Cluster Count vs Data Size")
                    axes[0,1].grid(True, alpha=0.3)
                    
                    # Plot 3: Performance vs percentage
                    axes[1,0].plot(fractions * 100, scores, 'o-', color='red', linewidth=2, markersize=6)
                    axes[1,0].set_xlabel("Data Used (%)")
                    axes[1,0].set_ylabel(metric_type)
                    axes[1,0].set_title(f"{metric_type} vs Data Percentage")
                    axes[1,0].grid(True, alpha=0.3)
                    
                    # Plot 4: Stability over data sizes
                    if stability_scores:
                        axes[1,1].plot(data_sizes[1:], stability_scores, 'o-', color='purple', linewidth=2, markersize=6)
                        axes[1,1].set_xlabel("Training Set Size")
                        axes[1,1].set_ylabel("Score Change")
                        axes[1,1].set_title("Performance Stability")
                        axes[1,1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Analysis
                    performance_improvement = scores[-1] - scores[0]
                    diminishing_returns = len(scores) > 2 and abs(scores[-1] - scores[-2]) < 0.01
                    
                    st.subheader("Unsupervised Data Analysis")
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric(f"Performance Improvement", f"{performance_improvement:.4f}")
                    
                    with col_b:
                        cluster_stability = np.std(cluster_counts)
                        st.metric("Cluster Count Std", f"{cluster_stability:.2f}")
                    
                    with col_c:
                        if stability_scores:
                            avg_stability = np.mean(stability_scores)
                            st.metric("Avg Score Change", f"{avg_stability:.4f}")
                    
                    # Recommendations
                    if performance_improvement > 0.05:
                        st.success(f"✅ Significant improvement with more data (+{performance_improvement:.3f})")
                        if diminishing_returns:
                            st.info("📊 Returns are diminishing - current data size may be sufficient")
                        else:
                            st.info("📈 Consider collecting more data for further improvements")
                    else:
                        st.info("Current data size appears sufficient for this unsupervised model")
                    
                    if cluster_stability < 1.0:
                        st.success("✅ Stable clustering structure across different data sizes")
                    else:
                        st.warning("⚠️ Cluster count varies significantly with data size")
        
        with col2:
            st.markdown("### Unsupervised Guidelines")
            st.info("""
            **Data Requirements:**
            
            **Silhouette Score:**
            - Look for plateau indicating optimal size
            - More data usually helps up to a point
            
            **Fraud Detection:**
            - More data can improve pattern detection
            - Monitor cluster quality vs fraud isolation
            
            **Cluster Stability:**
            - Consistent performance indicates good size
            - Large variations suggest more data needed
            
            **General Strategy:**
            - Balance quality metrics with stability
            - Consider computational costs
            """)

def _render_feature_learning_analysis():
    """Render feature learning analysis."""
    st.subheader("Feature Learning Analysis")
    
    st.markdown("""
    ### Feature Importance & Learning Insights
    
    Understand which features models consider most important
    and how feature importance changes across different model types (supervised and unsupervised).
    """)
    
    # Model type selection
    model_type = st.selectbox(
        "Select model type for feature analysis",
        options=["Supervised", "Unsupervised", "Comparison"],
        key="fi_model_type"
    )
    
    if model_type == "Supervised":
        supervised_models = get_supervised_models()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Multi-model feature importance comparison
            selected_models = st.multiselect(
                "Select supervised models for feature comparison",
                options=list(supervised_models.keys()),
                default=list(supervised_models.keys())[:3],
                key="fi_models"
            )
            
            if st.button("Analyze Supervised Feature Importances", key="fi_btn"):
                if selected_models:
                    X, y = load_data(use_synthetic=True)
                    
                    feature_importances = {}
                    
                    for model_name in selected_models:
                        model = clone(supervised_models[model_name])
                        model.fit(X, y)
                        
                        if hasattr(model, "feature_importances_"):
                            importances = model.feature_importances_
                        elif hasattr(model, "coef_"):
                            importances = np.abs(model.coef_).ravel()
                        else:
                            st.warning(f"{model_name} does not expose feature importances")
                            continue
                        
                        feature_importances[model_name] = importances
                    
                    if feature_importances:
                        # Create comparison plot
                        n_features = len(list(feature_importances.values())[0])
                        feature_indices = range(n_features)
                        
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
                        # Plot 1: Feature importance comparison
                        width = 0.8 / len(selected_models)
                        for i, (model_name, importances) in enumerate(feature_importances.items()):
                            positions = [x + i * width for x in feature_indices]
                            ax1.bar(positions, importances, width, label=model_name, alpha=0.8)
                        
                        ax1.set_xlabel("Feature Index")
                        ax1.set_ylabel("Importance")
                        ax1.set_title("Supervised Feature Importance Comparison")
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)
                        
                        # Plot 2: Top features heatmap
                        if len(selected_models) > 1:
                            import pandas as pd
                            importance_df = pd.DataFrame(feature_importances, index=[f"Feature_{i}" for i in range(n_features)])
                            
                            # Show top 20 features
                            avg_importance = importance_df.mean(axis=1)
                            top_features = avg_importance.nlargest(min(20, n_features))
                            top_df = importance_df.loc[top_features.index]
                            
                            im = ax2.imshow(top_df.values, cmap='YlOrRd', aspect='auto')
                            ax2.set_xticks(range(len(selected_models)))
                            ax2.set_xticklabels(selected_models, rotation=45)
                            ax2.set_yticks(range(len(top_features)))
                            ax2.set_yticklabels(top_features.index)
                            ax2.set_title("Top Features Heatmap")
                            
                            # Add colorbar
                            plt.colorbar(im, ax=ax2)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Feature ranking analysis
                        st.subheader("Supervised Feature Ranking Analysis")
                        if len(selected_models) > 1:
                            avg_importance = importance_df.mean(axis=1).sort_values(ascending=False)
                            top_10 = avg_importance.head(10)
                            
                            st.write("**Top 10 Most Important Features (Average across models):**")
                            for i, (feature, importance) in enumerate(top_10.items(), 1):
                                st.write(f"{i}. {feature}: {importance:.4f}")
        
        with col2:
            st.markdown("### Supervised Feature Analysis")
            st.info("""
            **Interpretation Guidelines:**
            
            **High Importance:**
            - Critical for model decisions
            - Focus data quality efforts here
            
            **Consistent Across Models:**
            - Robust important features
            - High confidence in relevance
            
            **Model-Specific:**
            - Different models use different strategies
            - Consider ensemble approaches
            """)
    
    elif model_type == "Unsupervised":
        unsupervised_models = get_unsupervised_models()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_models = st.multiselect(
                "Select unsupervised models for feature analysis",
                options=list(unsupervised_models.keys()),
                default=list(unsupervised_models.keys())[:3],
                key="fi_unsup_models"
            )
            
            if st.button("Analyze Unsupervised Feature Patterns", key="fi_unsup_btn"):
                if selected_models:
                    X, y = load_data(use_synthetic=True)
                    
                    # Preprocess data
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    feature_analysis = {}
                    
                    for model_name in selected_models:
                        model = clone(unsupervised_models[model_name])
                        labels = model.fit_predict(X_scaled)
                        
                        # Analyze feature importance through cluster centers or components
                        if hasattr(model, 'cluster_centers_'):
                            # For clustering models with centers
                            centers = model.cluster_centers_
                            # Calculate feature variance across cluster centers
                            feature_variance = np.var(centers, axis=0)
                            feature_analysis[model_name] = feature_variance
                        elif hasattr(model, 'components_'):
                            # For dimensionality reduction or mixture models
                            components = model.components_
                            # Calculate average absolute component values
                            feature_importance = np.mean(np.abs(components), axis=0)
                            feature_analysis[model_name] = feature_importance
                        else:
                            # For other models, calculate feature discrimination ability
                            unique_labels = set(labels)
                            if len(unique_labels) > 1:
                                feature_discrimination = []
                                for i in range(X_scaled.shape[1]):
                                    feature_values = X_scaled[:, i]
                                    cluster_means = []
                                    for cluster_id in unique_labels:
                                        if cluster_id != -1:
                                            cluster_mask = labels == cluster_id
                                            if cluster_mask.sum() > 0:
                                                cluster_mean = feature_values[cluster_mask].mean()
                                                cluster_means.append(cluster_mean)
                                    
                                    # Calculate variance of cluster means
                                    if len(cluster_means) > 1:
                                        discrimination = np.var(cluster_means)
                                    else:
                                        discrimination = 0
                                    feature_discrimination.append(discrimination)
                                
                                feature_analysis[model_name] = np.array(feature_discrimination)
                    
                    if feature_analysis:
                        # Create visualizations
                        n_features = len(list(feature_analysis.values())[0])
                        feature_indices = range(n_features)
                        
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
                        # Plot 1: Feature pattern comparison
                        for model_name, patterns in feature_analysis.items():
                            ax1.plot(feature_indices, patterns, 'o-', label=model_name, linewidth=2, alpha=0.7)
                        
                        ax1.set_xlabel("Feature Index")
                        ax1.set_ylabel("Pattern Strength")
                        ax1.set_title("Unsupervised Feature Pattern Analysis")
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)
                        
                        # Plot 2: Feature pattern heatmap
                        if len(selected_models) > 1:
                            import pandas as pd
                            pattern_df = pd.DataFrame(feature_analysis, index=[f"Feature_{i}" for i in range(n_features)])
                            
                            # Normalize patterns for better visualization
                            pattern_df_norm = pattern_df.div(pattern_df.max(axis=0), axis=1)
                            
                            # Show top features by average pattern strength
                            avg_pattern = pattern_df_norm.mean(axis=1)
                            top_features = avg_pattern.nlargest(min(20, n_features))
                            top_df = pattern_df_norm.loc[top_features.index]
                            
                            im = ax2.imshow(top_df.values, cmap='viridis', aspect='auto')
                            ax2.set_xticks(range(len(selected_models)))
                            ax2.set_xticklabels(selected_models, rotation=45)
                            ax2.set_yticks(range(len(top_features)))
                            ax2.set_yticklabels(top_features.index)
                            ax2.set_title("Top Feature Patterns Heatmap")
                            
                            # Add colorbar
                            plt.colorbar(im, ax=ax2)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Feature pattern analysis
                        st.subheader("Unsupervised Feature Pattern Analysis")
                        if len(selected_models) > 1:
                            avg_pattern = pattern_df.mean(axis=1).sort_values(ascending=False)
                            top_10 = avg_pattern.head(10)
                            
                            st.write("**Top 10 Most Discriminative Features (Average across models):**")
                            for i, (feature, strength) in enumerate(top_10.items(), 1):
                                st.write(f"{i}. {feature}: {strength:.4f}")
        
        with col2:
            st.markdown("### Unsupervised Feature Analysis")
            st.info("""
            **Pattern Types:**
            
            **Cluster Centers:**
            - Features with high variance across centers
            - Important for cluster separation
            
            **Components:**
            - Features with high component loadings
            - Capture main data variations
            
            **Discrimination:**
            - Features that separate clusters well
            - Show distinct patterns per cluster
            
            **Usage:**
            - Identify key features for dimensionality reduction
            - Focus data collection efforts
            - Understand data structure
            """)
    
    else:  # Comparison
        supervised_models = get_supervised_models()
        unsupervised_models = get_unsupervised_models()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_supervised = st.multiselect(
                "Select supervised models",
                options=list(supervised_models.keys()),
                default=list(supervised_models.keys())[:2],
                key="fi_comp_supervised"
            )
            
            selected_unsupervised = st.multiselect(
                "Select unsupervised models",
                options=list(unsupervised_models.keys()),
                default=list(unsupervised_models.keys())[:2],
                key="fi_comp_unsupervised"
            )
            
            if st.button("Compare Feature Learning Approaches", key="fi_comp_btn"):
                if selected_supervised or selected_unsupervised:
                    X, y = load_data(use_synthetic=True)
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    all_feature_analysis = {}
                    
                    # Analyze supervised models
                    for model_name in selected_supervised:
                        model = clone(supervised_models[model_name])
                        model.fit(X, y)
                        
                        if hasattr(model, "feature_importances_"):
                            importances = model.feature_importances_
                        elif hasattr(model, "coef_"):
                            importances = np.abs(model.coef_).ravel()
                        else:
                            continue
                        
                        all_feature_analysis[f"{model_name} (Supervised)"] = importances
                    
                    # Analyze unsupervised models
                    for model_name in selected_unsupervised:
                        model = clone(unsupervised_models[model_name])
                        labels = model.fit_predict(X_scaled)
                        
                        if hasattr(model, 'cluster_centers_'):
                            centers = model.cluster_centers_
                            feature_variance = np.var(centers, axis=0)
                            all_feature_analysis[f"{model_name} (Unsupervised)"] = feature_variance
                        elif hasattr(model, 'components_'):
                            components = model.components_
                            feature_importance = np.mean(np.abs(components), axis=0)
                            all_feature_analysis[f"{model_name} (Unsupervised)"] = feature_importance
                    
                    if all_feature_analysis:
                        # Normalize all feature importances for comparison
                        import pandas as pd
                        n_features = len(list(all_feature_analysis.values())[0])
                        
                        normalized_analysis = {}
                        for model_name, importances in all_feature_analysis.items():
                            # Normalize to 0-1 range
                            norm_importances = (importances - importances.min()) / (importances.max() - importances.min() + 1e-8)
                            normalized_analysis[model_name] = norm_importances
                        
                        comparison_df = pd.DataFrame(normalized_analysis, index=[f"Feature_{i}" for i in range(n_features)])
                        
                        # Create comparison visualization
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                        
                        # Plot 1: Feature importance comparison
                        feature_indices = range(n_features)
                        colors = ['blue' if 'Supervised' in name else 'red' for name in comparison_df.columns]
                        
                        for i, (model_name, importances) in enumerate(comparison_df.items()):
                            ax1.plot(feature_indices, importances, 'o-', label=model_name, 
                                   linewidth=2, alpha=0.7, color=colors[i])
                        
                        ax1.set_xlabel("Feature Index")
                        ax1.set_ylabel("Normalized Importance/Pattern Strength")
                        ax1.set_title("Supervised vs Unsupervised Feature Learning")
                        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        ax1.grid(True, alpha=0.3)
                        
                        # Plot 2: Top features heatmap
                        # Show top 15 features by average importance
                        avg_importance = comparison_df.mean(axis=1)
                        top_features = avg_importance.nlargest(min(15, n_features))
                        top_df = comparison_df.loc[top_features.index]
                        
                        im = ax2.imshow(top_df.values, cmap='RdYlBu_r', aspect='auto')
                        ax2.set_xticks(range(len(comparison_df.columns)))
                        ax2.set_xticklabels(comparison_df.columns, rotation=45, ha='right')
                        ax2.set_yticks(range(len(top_features)))
                        ax2.set_yticklabels(top_features.index)
                        ax2.set_title("Top Features: Supervised vs Unsupervised")
                        
                        plt.colorbar(im, ax=ax2)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Comparison analysis
                        st.subheader("Feature Learning Comparison")
                        
                        # Find features that are important in both approaches
                        supervised_cols = [col for col in comparison_df.columns if 'Supervised' in col]
                        unsupervised_cols = [col for col in comparison_df.columns if 'Unsupervised' in col]
                        
                        if supervised_cols and unsupervised_cols:
                            avg_supervised = comparison_df[supervised_cols].mean(axis=1)
                            avg_unsupervised = comparison_df[unsupervised_cols].mean(axis=1)
                            
                            # Find consensus features (high importance in both)
                            consensus_threshold = 0.5
                            consensus_features = avg_supervised[(avg_supervised > consensus_threshold) & 
                                                              (avg_unsupervised > consensus_threshold)].sort_values(ascending=False)
                            
                            if len(consensus_features) > 0:
                                st.success(f"**Consensus Features (Important in both approaches):**")
                                for i, (feature, _) in enumerate(consensus_features.head(5).items(), 1):
                                    st.write(f"{i}. {feature}")
                            
                            # Find approach-specific features
                            supervised_specific = avg_supervised[(avg_supervised > consensus_threshold) & 
                                                               (avg_unsupervised <= consensus_threshold)].sort_values(ascending=False)
                            unsupervised_specific = avg_unsupervised[(avg_unsupervised > consensus_threshold) & 
                                                                   (avg_supervised <= consensus_threshold)].sort_values(ascending=False)
                            
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                if len(supervised_specific) > 0:
                                    st.info("**Supervised-Specific Features:**")
                                    for i, (feature, _) in enumerate(supervised_specific.head(3).items(), 1):
                                        st.write(f"{i}. {feature}")
                            
                            with col_b:
                                if len(unsupervised_specific) > 0:
                                    st.info("**Unsupervised-Specific Features:**")
                                    for i, (feature, _) in enumerate(unsupervised_specific.head(3).items(), 1):
                                        st.write(f"{i}. {feature}")
        
        with col2:
            st.markdown("### Comparison Insights")
            st.info("""
            **Feature Learning Differences:**
            
            **Supervised Learning:**
            - Focuses on discriminative features
            - Optimizes for prediction accuracy
            - May ignore irrelevant patterns
            
            **Unsupervised Learning:**
            - Discovers data structure
            - Finds natural groupings
            - May reveal hidden patterns
            
            **Consensus Features:**
            - Important across approaches
            - High confidence features
            - Prime candidates for feature selection
            
            **Approach-Specific:**
            - Reveals different perspectives
            - Consider for ensemble methods
            - Validate with domain knowledge
            """)
