import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

from model_utils import get_supervised_models, get_unsupervised_models, load_data

def render_robustness_testing_tab():
    """Render robustness testing tab."""
    st.header("🔍 Robustness Testing")
    
    st.markdown("""
    ### Model Robustness & Reliability Assessment
    
    Evaluate how well models perform under various challenging conditions including
    adversarial attacks, data drift, and edge cases for both supervised and unsupervised models.
    """)
    
    robust_tabs = st.tabs([
        "Adversarial Testing",
        "Data Perturbation",
        "Edge Case Analysis",
        "Concept Drift Detection",
        "Stability Assessment"
    ])
    
    with robust_tabs[0]:
        _render_adversarial_testing()
    
    with robust_tabs[1]:
        _render_data_perturbation_testing()
    
    with robust_tabs[2]:
        _render_edge_case_analysis()
    
    with robust_tabs[3]:
        _render_concept_drift_detection()
    
    with robust_tabs[4]:
        _render_stability_assessment()

def _render_data_perturbation_testing():
    """Render data perturbation testing."""
    st.subheader("Data Perturbation Testing")
    
    st.markdown("""
    ### Model Sensitivity to Input Variations
    
    Test how models respond to various types of realistic data perturbations
    that might occur in production environments.
    """)
    
    # Model type selection
    model_type = st.selectbox(
        "Select model type for perturbation testing",
        options=["Supervised", "Unsupervised"],
        key="pert_model_type"
    )
    
    perturbation_type = st.selectbox(
        "Select perturbation type",
        options=["Gaussian Noise", "Feature Dropout", "Feature Scaling", "Outlier Injection"],
        key="pert_type"
    )
    
    if model_type == "Supervised":
        supervised_models = get_supervised_models()
        selected_model = st.selectbox(
            "Select supervised model for testing",
            options=list(supervised_models.keys()),
            key="pert_supervised_model"
        )
        models_dict = supervised_models
    else:
        unsupervised_models = get_unsupervised_models()
        selected_model = st.selectbox(
            "Select unsupervised model for testing",
            options=list(unsupervised_models.keys()),
            key="pert_unsupervised_model"
        )
        models_dict = unsupervised_models
    
    if st.button("Run Perturbation Testing", key="pert_btn"):
        X, y = load_data(use_synthetic=True)
        
        if model_type == "Supervised":
            _run_supervised_perturbation_test(X, y, models_dict[selected_model], selected_model, perturbation_type)
        else:
            _run_unsupervised_perturbation_test(X, y, models_dict[selected_model], selected_model, perturbation_type)

def _render_adversarial_testing():
    """Render adversarial testing analysis."""
    st.subheader("Adversarial Testing")
    
    st.markdown("""
    ### Model Robustness Against Adversarial Attacks
    
    Test how models perform when faced with carefully crafted inputs designed
    to fool the classifier while remaining imperceptible to humans.
    """)
    
    # Model type selection
    model_type = st.selectbox(
        "Select model type for adversarial testing",
        options=["Supervised", "Unsupervised"],
        key="adv_model_type"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if model_type == "Supervised":
            supervised_models = get_supervised_models()
            selected_model = st.selectbox(
                "Select supervised model for adversarial testing",
                options=list(supervised_models.keys()),
                key="adv_supervised_model"
            )
            models_dict = supervised_models
        else:
            unsupervised_models = get_unsupervised_models()
            selected_model = st.selectbox(
                "Select unsupervised model for adversarial testing",
                options=list(unsupervised_models.keys()),
                key="adv_unsupervised_model"
            )
            models_dict = unsupervised_models
        
        attack_strength = st.slider(
            "Attack strength (epsilon)",
            0.01, 0.5, 0.1, 0.01,
            key="attack_strength"
        )
        
        if st.button("Run Adversarial Testing", key="adv_btn"):
            X, y = load_data(use_synthetic=True)
            
            if model_type == "Supervised":
                _run_supervised_adversarial_test(X, y, models_dict[selected_model], selected_model, attack_strength)
            else:
                _run_unsupervised_adversarial_test(X, y, models_dict[selected_model], selected_model, attack_strength)
    
    with col2:
        st.markdown("### Robustness Guidelines")
        st.info("""
        **Robustness Interpretation:**
        
        **Supervised Models:**
        - High Robustness (>0.9): Stable predictions
        - Medium Robustness (0.7-0.9): Acceptable
        - Low Robustness (<0.7): Vulnerable to attacks
        
        **Unsupervised Models:**
        - Stable clustering structure
        - Consistent cluster assignments
        - Maintained separation quality
        """)

def _render_edge_case_analysis():
    """Render edge case analysis."""
    st.subheader("Edge Case Analysis")
    
    st.markdown("""
    ### Extreme and Boundary Condition Testing
    
    Analyze model behavior on edge cases and extreme values that may occur
    in real-world scenarios but are rare in training data.
    """)
    
    # Model type selection
    model_type = st.selectbox(
        "Select model type for edge case analysis",
        options=["Supervised", "Unsupervised"],
        key="edge_model_type"
    )
    
    edge_case_type = st.selectbox(
        "Select edge case type",
        options=["Extreme Values", "Zero Values", "Missing Features", "Rare Combinations"],
        key="edge_type"
    )
    
    if st.button("Analyze Edge Cases", key="edge_btn"):
        X, y = load_data(use_synthetic=True)
        
        if model_type == "Supervised":
            _analyze_supervised_edge_cases(X, y, edge_case_type)
        else:
            _analyze_unsupervised_edge_cases(X, y, edge_case_type)

def _render_concept_drift_detection():
    """Render concept drift detection analysis."""
    st.subheader("Concept Drift Detection")
    
    st.markdown("""
    ### Monitoring for Changes in Data Distribution
    
    Detect when the relationship between features and target changes over time,
    which can degrade model performance in production.
    """)
    
    # Simulate concept drift
    if st.button("Simulate Concept Drift Analysis", key="drift_btn"):
        X, y = load_data(use_synthetic=True)
        
        # Split data into time periods
        n_periods = 5
        period_size = len(X) // n_periods
        
        supervised_models = get_supervised_models()
        selected_model = list(supervised_models.keys())[0]  # Use first model
        model = supervised_models[selected_model]
        
        # Train on first period
        X_train_initial = X[:period_size]
        y_train_initial = y[:period_size]
        model.fit(X_train_initial, y_train_initial)
        
        # Test on subsequent periods with simulated drift
        period_scores = []
        drift_magnitudes = []
        
        for i in range(1, n_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size
            
            X_period = X[start_idx:end_idx]
            y_period = y[start_idx:end_idx]
            
            # Simulate drift by adding increasing noise
            drift_magnitude = i * 0.1
            X_period_drift = X_period + np.random.normal(0, drift_magnitude, X_period.shape)
            
            # Evaluate performance
            score = model.score(X_period_drift, y_period)
            period_scores.append(score)
            drift_magnitudes.append(drift_magnitude)
        
        # Plot drift analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        periods = list(range(1, n_periods))
        
        # Performance over time
        ax1.plot(periods, period_scores, 'ro-', linewidth=2, markersize=8)
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Model Accuracy')
        ax1.set_title('Model Performance Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Drift magnitude vs performance
        ax2.scatter(drift_magnitudes, period_scores, s=100, alpha=0.7)
        ax2.set_xlabel('Drift Magnitude')
        ax2.set_ylabel('Model Accuracy')
        ax2.set_title('Performance vs Drift Magnitude')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Drift detection alerts
        baseline_score = period_scores[0]
        current_score = period_scores[-1]
        performance_drop = (baseline_score - current_score) / baseline_score * 100
        
        if performance_drop > 10:
            st.error(f"🚨 Significant drift detected! Performance dropped by {performance_drop:.1f}%")
        elif performance_drop > 5:
            st.warning(f"⚠️ Moderate drift detected. Performance dropped by {performance_drop:.1f}%")
        else:
            st.success("✅ No significant drift detected")

def _render_stability_assessment():
    """Render model stability assessment."""
    st.subheader("Model Stability Assessment")
    
    st.markdown("""
    ### Evaluating Prediction Consistency
    
    Assess how stable model predictions are across different conditions
    and whether small changes lead to large prediction differences for both supervised and unsupervised models.
    """)
    
    # Model type selection
    model_type = st.selectbox(
        "Select model type for stability testing",
        options=["Supervised", "Unsupervised"],
        key="stability_model_type"
    )
    
    if model_type == "Supervised":
        supervised_models = get_supervised_models()
        selected_model = st.selectbox(
            "Select supervised model for stability testing",
            options=list(supervised_models.keys()),
            key="robust_stability_model"
        )
        
        if st.button("Run Supervised Stability Assessment", key="robustness_stability_btn"):
            X, y = load_data(use_synthetic=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            model = supervised_models[selected_model]
            model.fit(X_train, y_train)
            
            # Test 1: Prediction consistency with small perturbations
            n_tests = 100
            stability_scores = []
            
            for i in range(n_tests):
                # Add small random noise
                noise = np.random.normal(0, 0.01, X_test.shape)
                X_perturbed = X_test + noise
                
                # Get predictions
                orig_pred = model.predict(X_test)
                pert_pred = model.predict(X_perturbed)
                
                # Calculate agreement
                agreement = np.mean(orig_pred == pert_pred)
                stability_scores.append(agreement)
            
            avg_stability = np.mean(stability_scores)
            stability_std = np.std(stability_scores)
            
            # Test 2: Confidence stability
            if hasattr(model, 'predict_proba'):
                confidence_stabilities = []
                
                for i in range(n_tests):
                    noise = np.random.normal(0, 0.01, X_test.shape)
                    X_perturbed = X_test + noise
                    
                    orig_conf = model.predict_proba(X_test)[:, 1]
                    pert_conf = model.predict_proba(X_perturbed)[:, 1]
                    
                    # Calculate confidence correlation
                    conf_corr = np.corrcoef(orig_conf, pert_conf)[0, 1]
                    confidence_stabilities.append(conf_corr)
                
                avg_conf_stability = np.mean(confidence_stabilities)
            else:
                avg_conf_stability = np.nan
            
            # Results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prediction Stability", f"{avg_stability:.4f}")
            with col2:
                st.metric("Stability Std Dev", f"{stability_std:.4f}")
            with col3:
                if not np.isnan(avg_conf_stability):
                    st.metric("Confidence Stability", f"{avg_conf_stability:.4f}")
                else:
                    st.metric("Confidence Stability", "N/A")
            
            # Visualizations
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Stability distribution
            ax1.hist(stability_scores, bins=20, alpha=0.7, edgecolor='black')
            ax1.axvline(avg_stability, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_stability:.3f}')
            ax1.set_xlabel('Prediction Stability')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Distribution of Stability Scores')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Stability over iterations
            ax2.plot(stability_scores, alpha=0.7)
            ax2.axhline(avg_stability, color='red', linestyle='--', label=f'Average: {avg_stability:.3f}')
            ax2.set_xlabel('Test Iteration')
            ax2.set_ylabel('Stability Score')
            ax2.set_title('Stability Over Test Iterations')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Interpretation
            if avg_stability > 0.95:
                st.success("✅ Excellent stability - Model predictions are highly consistent")
            elif avg_stability > 0.90:
                st.info("✓ Good stability - Model is reasonably stable")
            elif avg_stability > 0.85:
                st.warning("⚠️ Moderate stability - Consider regularization or ensemble methods")
            else:
                st.error("❌ Poor stability - Model predictions are inconsistent")
    
    else:  # Unsupervised
        unsupervised_models = get_unsupervised_models()
        selected_model = st.selectbox(
            "Select unsupervised model for stability testing",
            options=list(unsupervised_models.keys()),
            key="unsup_stability_model"
        )
        
        if st.button("Run Unsupervised Stability Assessment", key="unsup_stability_btn"):
            X, y = load_data(use_synthetic=True)
            
            # Preprocess data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = unsupervised_models[selected_model]
            
            # Test 1: Clustering consistency with small perturbations
            n_tests = 50  # Fewer tests for unsupervised due to computational cost
            stability_scores = []
            silhouette_scores = []
            cluster_count_consistency = []
            
            # Get original clustering
            original_labels = model.fit_predict(X_scaled)
            original_n_clusters = len(set(original_labels)) - (1 if -1 in original_labels else 0)
            original_silhouette = silhouette_score(X_scaled, original_labels) if original_n_clusters > 1 else 0
            
            progress_bar = st.progress(0)
            
            for i in range(n_tests):
                # Add small random noise
                noise = np.random.normal(0, 0.01, X_scaled.shape)
                X_perturbed = X_scaled + noise
                
                # Get perturbed clustering
                perturbed_labels = model.fit_predict(X_perturbed)
                perturbed_n_clusters = len(set(perturbed_labels)) - (1 if -1 in perturbed_labels else 0)
                
                # Calculate clustering stability (ARI)
                stability = adjusted_rand_score(original_labels, perturbed_labels)
                stability_scores.append(stability)
                
                # Calculate silhouette score stability
                if perturbed_n_clusters > 1:
                    perturbed_silhouette = silhouette_score(X_perturbed, perturbed_labels)
                    silhouette_scores.append(perturbed_silhouette)
                else:
                    silhouette_scores.append(0)
                
                # Track cluster count consistency
                cluster_count_consistency.append(perturbed_n_clusters == original_n_clusters)
                
                progress_bar.progress((i + 1) / n_tests)
            
            # Calculate metrics
            avg_stability = np.mean(stability_scores)
            stability_std = np.std(stability_scores)
            avg_silhouette_perturbed = np.mean(silhouette_scores)
            silhouette_stability = avg_silhouette_perturbed / max(original_silhouette, 0.001)
            cluster_consistency = np.mean(cluster_count_consistency)
            
            # Results
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Clustering Stability (ARI)", f"{avg_stability:.4f}")
            with col2:
                st.metric("Stability Std Dev", f"{stability_std:.4f}")
            with col3:
                st.metric("Silhouette Stability", f"{silhouette_stability:.4f}")
            with col4:
                st.metric("Cluster Count Consistency", f"{cluster_consistency:.4f}")
            
            # Detailed metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Original Clusters", original_n_clusters)
            with col_b:
                st.metric("Original Silhouette", f"{original_silhouette:.4f}")
            with col_c:
                st.metric("Avg Perturbed Silhouette", f"{avg_silhouette_perturbed:.4f}")
            
            # Visualizations
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Stability distribution
            axes[0,0].hist(stability_scores, bins=20, alpha=0.7, edgecolor='black')
            axes[0,0].axvline(avg_stability, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_stability:.3f}')
            axes[0,0].set_xlabel('Clustering Stability (ARI)')
            axes[0,0].set_ylabel('Frequency')
            axes[0,0].set_title('Distribution of Clustering Stability')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            
            # Stability over iterations
            axes[0,1].plot(stability_scores, alpha=0.7, color='blue')
            axes[0,1].axhline(avg_stability, color='red', linestyle='--', label=f'Average: {avg_stability:.3f}')
            axes[0,1].set_xlabel('Test Iteration')
            axes[0,1].set_ylabel('Clustering Stability (ARI)')
            axes[0,1].set_title('Stability Over Test Iterations')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
            
            # Silhouette score comparison
            axes[1,0].hist(silhouette_scores, bins=20, alpha=0.7, edgecolor='black', color='green')
            axes[1,0].axvline(original_silhouette, color='red', linestyle='--', linewidth=2, label=f'Original: {original_silhouette:.3f}')
            axes[1,0].axvline(avg_silhouette_perturbed, color='blue', linestyle='--', linewidth=2, label=f'Avg Perturbed: {avg_silhouette_perturbed:.3f}')
            axes[1,0].set_xlabel('Silhouette Score')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].set_title('Silhouette Score Distribution')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            
            # Cluster count consistency
            consistency_counts = [np.sum(cluster_count_consistency), len(cluster_count_consistency) - np.sum(cluster_count_consistency)]
            labels = ['Consistent', 'Inconsistent']
            axes[1,1].pie(consistency_counts, labels=labels, autopct='%1.1f%%', startangle=90)
            axes[1,1].set_title('Cluster Count Consistency')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Interpretation for unsupervised models
            st.subheader("Unsupervised Stability Interpretation")
            
            if avg_stability > 0.8:
                st.success("✅ Excellent clustering stability - Consistent cluster assignments")
            elif avg_stability > 0.6:
                st.info("✓ Good clustering stability - Reasonably consistent clustering")
            elif avg_stability > 0.4:
                st.warning("⚠️ Moderate stability - Clustering varies with small perturbations")
            else:
                st.error("❌ Poor stability - Highly sensitive to input variations")
            
            if silhouette_stability > 0.9:
                st.success("✅ Silhouette score is stable across perturbations")
            elif silhouette_stability > 0.8:
                st.info("✓ Reasonable silhouette stability")
            else:
                st.warning("⚠️ Clustering quality varies significantly with perturbations")
            
            if cluster_consistency > 0.8:
                st.success("✅ Consistent number of clusters detected")
            elif cluster_consistency > 0.6:
                st.info("✓ Mostly consistent cluster count")
            else:
                st.warning("⚠️ Number of clusters varies frequently - consider parameter tuning")

# Supporting functions for the tests
def _run_supervised_adversarial_test(X, y, model, model_name, attack_strength):
    """Run adversarial testing for supervised models."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    model.fit(X_train, y_train)
    
    # Simple adversarial attack simulation (FGSM-like)
    original_accuracy = model.score(X_test, y_test)
    
    # Generate adversarial examples
    np.random.seed(42)
    noise = np.random.normal(0, attack_strength, X_test.shape)
    X_adv = X_test + noise
    
    # Evaluate on adversarial examples
    adv_accuracy = model.score(X_adv, y_test)
    
    # Results
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Original Accuracy", f"{original_accuracy:.4f}")
    with col_b:
        st.metric("Adversarial Accuracy", f"{adv_accuracy:.4f}")
    with col_c:
        robustness = adv_accuracy / original_accuracy
        st.metric("Robustness Ratio", f"{robustness:.4f}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    categories = ['Original', 'Adversarial']
    accuracies = [original_accuracy, adv_accuracy]
    ax1.bar(categories, accuracies, color=['blue', 'red'], alpha=0.7)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy: Original vs Adversarial')
    ax1.set_ylim(0, 1)
    
    # Prediction confidence distribution
    if hasattr(model, 'predict_proba'):
        orig_conf = model.predict_proba(X_test)[:, 1]
        adv_conf = model.predict_proba(X_adv)[:, 1]
        
        ax2.hist(orig_conf, alpha=0.5, label='Original', bins=20)
        ax2.hist(adv_conf, alpha=0.5, label='Adversarial', bins=20)
        ax2.set_xlabel('Prediction Confidence')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Confidence Distribution')
        ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)

def _run_unsupervised_adversarial_test(X, y, model, model_name, attack_strength):
    """Run adversarial testing for unsupervised models."""
    # Preprocess data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Original clustering
    original_labels = model.fit_predict(X_scaled)
    
    # Generate adversarial examples
    np.random.seed(42)
    noise = np.random.normal(0, attack_strength, X_scaled.shape)
    X_adv = X_scaled + noise
    
    # Adversarial clustering
    adv_labels = model.fit_predict(X_adv)
    
    # Calculate robustness metrics
    original_n_clusters = len(set(original_labels)) - (1 if -1 in original_labels else 0)
    adv_n_clusters = len(set(adv_labels)) - (1 if -1 in adv_labels else 0)
    
    # Silhouette scores
    original_silhouette = silhouette_score(X_scaled, original_labels) if original_n_clusters > 1 else 0
    adv_silhouette = silhouette_score(X_adv, adv_labels) if adv_n_clusters > 1 else 0
    
    # Cluster stability (how many points remain in same cluster)
    cluster_stability = adjusted_rand_score(original_labels, adv_labels)
    
    # Results
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Original Clusters", original_n_clusters)
    with col_b:
        st.metric("Adversarial Clusters", adv_n_clusters)
    with col_c:
        st.metric("Silhouette Robustness", f"{adv_silhouette/max(original_silhouette, 0.001):.4f}")
    with col_d:
        st.metric("Cluster Stability (ARI)", f"{cluster_stability:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original vs adversarial silhouette scores
    axes[0,0].bar(['Original', 'Adversarial'], [original_silhouette, adv_silhouette], 
                  color=['blue', 'red'], alpha=0.7)
    axes[0,0].set_ylabel('Silhouette Score')
    axes[0,0].set_title('Clustering Quality: Original vs Adversarial')
    
    # Number of clusters comparison
    axes[0,1].bar(['Original', 'Adversarial'], [original_n_clusters, adv_n_clusters], 
                  color=['blue', 'red'], alpha=0.7)
    axes[0,1].set_ylabel('Number of Clusters')
    axes[0,1].set_title('Cluster Count: Original vs Adversarial')
    
    # Cluster size distributions (if available)
    if original_n_clusters > 0:
        orig_sizes = [np.sum(original_labels == i) for i in set(original_labels) if i != -1]
        axes[1,0].hist(orig_sizes, alpha=0.7, label='Original', bins=min(10, len(orig_sizes)))
        axes[1,0].set_xlabel('Cluster Size')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Original Cluster Size Distribution')
    
    if adv_n_clusters > 0:
        adv_sizes = [np.sum(adv_labels == i) for i in set(adv_labels) if i != -1]
        axes[1,1].hist(adv_sizes, alpha=0.7, label='Adversarial', bins=min(10, len(adv_sizes)), color='red')
        axes[1,1].set_xlabel('Cluster Size')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Adversarial Cluster Size Distribution')
    
    plt.tight_layout()
    st.pyplot(fig)

def _run_supervised_perturbation_test(X, y, model, model_name, perturbation_type):
    """Run perturbation testing for supervised models."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    model.fit(X_train, y_train)
    
    # Original performance
    original_score = model.score(X_test, y_test)
    
    # Test different perturbation levels
    perturbation_levels = np.linspace(0.0, 0.5, 11)
    scores = []
    
    for level in perturbation_levels:
        X_pert = _apply_perturbation(X_test, perturbation_type, level)
        score = model.score(X_pert, y_test)
        scores.append(score)
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(perturbation_levels, scores, 'bo-', linewidth=2, markersize=6)
    ax.axhline(y=original_score, color='r', linestyle='--', label='Original Score')
    ax.set_xlabel('Perturbation Level')
    ax.set_ylabel('Model Accuracy')
    ax.set_title(f'Supervised Model Robustness to {perturbation_type}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Summary
    final_score = scores[-1]
    degradation = (original_score - final_score) / original_score * 100
    st.metric("Performance Degradation", f"{degradation:.1f}%")

def _run_unsupervised_perturbation_test(X, y, model, model_name, perturbation_type):
    """Run perturbation testing for unsupervised models."""
    # Preprocess data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Original clustering performance
    original_labels = model.fit_predict(X_scaled)
    original_silhouette = silhouette_score(X_scaled, original_labels) if len(set(original_labels)) > 1 else 0
    
    # Test different perturbation levels
    perturbation_levels = np.linspace(0.0, 0.5, 11)
    silhouette_scores = []
    cluster_counts = []
    stability_scores = []
    
    for level in perturbation_levels:
        X_pert = _apply_perturbation(X_scaled, perturbation_type, level)
        pert_labels = model.fit_predict(X_pert)
        
        # Calculate metrics
        n_clusters = len(set(pert_labels)) - (1 if -1 in pert_labels else 0)
        cluster_counts.append(n_clusters)
        
        if n_clusters > 1:
            sil_score = silhouette_score(X_pert, pert_labels)
            silhouette_scores.append(sil_score)
        else:
            silhouette_scores.append(0)
        
        # Stability compared to original
        stability = adjusted_rand_score(original_labels, pert_labels)
        stability_scores.append(stability)
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Silhouette scores
    axes[0].plot(perturbation_levels, silhouette_scores, 'bo-', linewidth=2, markersize=6)
    axes[0].axhline(y=original_silhouette, color='r', linestyle='--', label='Original Score')
    axes[0].set_xlabel('Perturbation Level')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].set_title(f'Clustering Quality vs {perturbation_type}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Cluster counts
    axes[1].plot(perturbation_levels, cluster_counts, 'go-', linewidth=2, markersize=6)
    axes[1].axhline(y=len(set(original_labels)) - (1 if -1 in original_labels else 0), 
                    color='r', linestyle='--', label='Original Count')
    axes[1].set_xlabel('Perturbation Level')
    axes[1].set_ylabel('Number of Clusters')
    axes[1].set_title(f'Cluster Count vs {perturbation_type}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Stability scores
    axes[2].plot(perturbation_levels, stability_scores, 'ro-', linewidth=2, markersize=6)
    axes[2].set_xlabel('Perturbation Level')
    axes[2].set_ylabel('ARI (Stability)')
    axes[2].set_title(f'Cluster Stability vs {perturbation_type}')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        final_silhouette = silhouette_scores[-1]
        sil_degradation = (original_silhouette - final_silhouette) / max(original_silhouette, 0.001) * 100
        st.metric("Silhouette Degradation", f"{sil_degradation:.1f}%")
    
    with col2:
        final_stability = stability_scores[-1]
        st.metric("Final Stability (ARI)", f"{final_stability:.4f}")
    
    with col3:
        cluster_stability = np.std(cluster_counts)
        st.metric("Cluster Count Variance", f"{cluster_stability:.2f}")

def _apply_perturbation(X, perturbation_type, level):
    """Apply perturbation to data based on type and level."""
    X_pert = X.copy()
    
    if perturbation_type == "Gaussian Noise":
        noise = np.random.normal(0, level, X.shape)
        X_pert = X + noise
    elif perturbation_type == "Feature Dropout":
        n_drop = int(level * X.shape[1])
        if n_drop > 0:
            drop_indices = np.random.choice(X.shape[1], n_drop, replace=False)
            X_pert[:, drop_indices] = 0
    elif perturbation_type == "Feature Scaling":
        X_pert = X * (1 + level)
    else:  # Outlier Injection
        n_outliers = int(level * X.shape[0])
        if n_outliers > 0:
            outlier_indices = np.random.choice(X.shape[0], n_outliers, replace=False)
            X_pert[outlier_indices] += np.random.normal(0, 3, X_pert[outlier_indices].shape)
    
    return X_pert

def _analyze_supervised_edge_cases(X, y, edge_case_type):
    """Analyze edge cases for supervised models."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    supervised_models = get_supervised_models()
    
    # Generate edge cases
    X_edge = _generate_edge_cases(X_test, edge_case_type)
    
    # Test all models on edge cases
    results = {}
    for model_name, model in supervised_models.items():
        try:
            model.fit(X_train, y_train)
            
            # Normal performance
            normal_score = model.score(X_test, y_test)
            
            # Edge case predictions
            edge_predictions = model.predict(X_edge)
            
            if hasattr(model, 'predict_proba'):
                edge_probs = model.predict_proba(X_edge)[:, 1]
                avg_confidence = np.mean(edge_probs)
            else:
                avg_confidence = np.nan
            
            results[model_name] = {
                'Normal_Accuracy': normal_score,
                'Edge_Predictions': edge_predictions,
                'Avg_Confidence': avg_confidence,
                'Fraud_Rate': np.mean(edge_predictions)
            }
        except Exception as e:
            st.warning(f"Model {model_name} failed on edge cases: {str(e)}")
    
    # Display results
    if results:
        results_df = pd.DataFrame({
            model: {
                'Normal Accuracy': data['Normal_Accuracy'],
                'Edge Case Fraud Rate': data['Fraud_Rate'],
                'Avg Confidence': data['Avg_Confidence']
            }
            for model, data in results.items()
        }).T
        
        st.subheader(f"Supervised Edge Case Results: {edge_case_type}")
        styled_edge = results_df.style.format({
            'Normal Accuracy': '{:.4f}',
            'Edge Case Fraud Rate': '{:.4f}',
            'Avg Confidence': '{:.4f}'
        })
        st.dataframe(styled_edge)

def _analyze_unsupervised_edge_cases(X, y, edge_case_type):
    """Analyze edge cases for unsupervised models."""
    # Preprocess data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    unsupervised_models = get_unsupervised_models()
    
    # Generate edge cases
    X_edge = _generate_edge_cases(X_scaled, edge_case_type)
    
    # Test all models on edge cases
    results = {}
    for model_name, model in unsupervised_models.items():
        try:
            # Normal clustering
            normal_labels = model.fit_predict(X_scaled)
            normal_n_clusters = len(set(normal_labels)) - (1 if -1 in normal_labels else 0)
            normal_silhouette = silhouette_score(X_scaled, normal_labels) if normal_n_clusters > 1 else 0
            
            # Edge case clustering
            edge_labels = model.fit_predict(X_edge)
            edge_n_clusters = len(set(edge_labels)) - (1 if -1 in edge_labels else 0)
            edge_silhouette = silhouette_score(X_edge, edge_labels) if edge_n_clusters > 1 else 0
            
            # Stability
            stability = adjusted_rand_score(normal_labels[:len(X_edge)], edge_labels) if len(X_edge) <= len(normal_labels) else np.nan
            
            results[model_name] = {
                'Normal_Clusters': normal_n_clusters,
                'Edge_Clusters': edge_n_clusters,
                'Normal_Silhouette': normal_silhouette,
                'Edge_Silhouette': edge_silhouette,
                'Stability': stability
            }
        except Exception as e:
            st.warning(f"Model {model_name} failed on edge cases: {str(e)}")
    
    # Display results
    if results:
        results_df = pd.DataFrame(results).T
        
        st.subheader(f"Unsupervised Edge Case Results: {edge_case_type}")
        styled_edge = results_df.style.format({
            'Normal_Silhouette': '{:.4f}',
            'Edge_Silhouette': '{:.4f}',
            'Stability': '{:.4f}'
        })
        st.dataframe(styled_edge)

def _generate_edge_cases(X, edge_case_type):
    """Generate edge cases based on type."""
    if edge_case_type == "Extreme Values":
        # Test with extreme values (5 standard deviations)
        X_edge = np.copy(X[:10])  # Take first 10 samples
        X_edge = X_edge * 5  # Scale to extreme values
        
    elif edge_case_type == "Zero Values":
        # Test with all zeros
        X_edge = np.zeros((10, X.shape[1]))
        
    elif edge_case_type == "Missing Features":
        # Test with missing features (set to mean)
        X_edge = np.copy(X[:10])
        # Randomly set 50% of features to mean (0 after standardization)
        mask = np.random.random(X_edge.shape) < 0.5
        X_edge[mask] = 0
        
    else:  # Rare Combinations
        # Create synthetic rare combinations
        X_edge = np.random.normal(0, 2, (10, X.shape[1]))
    
    return X_edge
