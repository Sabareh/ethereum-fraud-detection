import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.base import clone

from model_utils import get_supervised_models, get_unsupervised_models, load_data

def render_efficiency_analysis_tab():
    """Render efficiency analysis tab."""
    st.header("⚡ Efficiency Analysis")
    
    st.markdown("""
    ### Computational Efficiency & Resource Utilization
    
    Analyze model performance in terms of computational cost, memory usage,
    and scalability for production deployment.
    """)
    
    efficiency_tabs = st.tabs([
        "Training Efficiency",
        "Inference Speed",
        "Memory Analysis",
        "Scalability Testing",
        "Resource Optimization"
    ])
    
    with efficiency_tabs[0]:
        _render_training_efficiency()
    
    with efficiency_tabs[1]:
        _render_inference_speed_analysis()
    
    with efficiency_tabs[2]:
        _render_memory_analysis()
    
    with efficiency_tabs[3]:
        _render_scalability_testing()
    
    with efficiency_tabs[4]:
        _render_resource_optimization()

def _render_training_efficiency():
    """Render training efficiency analysis."""
    st.subheader("Training Efficiency")
    
    st.markdown("""
    ### Training Time vs Data Size Analysis
    
    Understand how training time scales with data size and compare
    efficiency across different models (both supervised and unsupervised).
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Model type selection
        model_type = st.selectbox(
            "Select model type",
            options=["Supervised", "Unsupervised", "Both"],
            key="te_model_type"
        )
        
        if model_type == "Supervised":
            supervised_models = get_supervised_models()
            selected_models = st.multiselect(
                "Select supervised models for training efficiency comparison",
                options=list(supervised_models.keys()),
                default=list(supervised_models.keys())[:3],
                key="te_supervised_models"
            )
            models_dict = {name: model for name, model in supervised_models.items() if name in selected_models}
        elif model_type == "Unsupervised":
            unsupervised_models = get_unsupervised_models()
            selected_models = st.multiselect(
                "Select unsupervised models for training efficiency comparison",
                options=list(unsupervised_models.keys()),
                default=list(unsupervised_models.keys())[:3],
                key="te_unsupervised_models"
            )
            models_dict = {name: model for name, model in unsupervised_models.items() if name in selected_models}
        else:  # Both
            supervised_models = get_supervised_models()
            unsupervised_models = get_unsupervised_models()
            
            selected_supervised = st.multiselect(
                "Select supervised models",
                options=list(supervised_models.keys()),
                default=list(supervised_models.keys())[:2],
                key="te_both_supervised"
            )
            selected_unsupervised = st.multiselect(
                "Select unsupervised models",
                options=list(unsupervised_models.keys()),
                default=list(unsupervised_models.keys())[:2],
                key="te_both_unsupervised"
            )
            
            models_dict = {}
            models_dict.update({f"{name} (S)": model for name, model in supervised_models.items() if name in selected_supervised})
            models_dict.update({f"{name} (U)": model for name, model in unsupervised_models.items() if name in selected_unsupervised})
            selected_models = list(models_dict.keys())
        
        if st.button("Analyze Training Efficiency", key="te_btn"):
            if selected_models:
                X, y = load_data(use_synthetic=True)
                fractions = np.linspace(0.2, 0.9, 5)
                
                training_times = {}
                
                progress_bar = st.progress(0)
                total_operations = len(selected_models) * len(fractions)
                current_op = 0
                
                for model_name in selected_models:
                    times = []
                    
                    for f in fractions:
                        X_sub, _, y_sub, _ = train_test_split(
                            X, y, train_size=f, random_state=42, stratify=y
                        )
                        
                        model = clone(models_dict[model_name])
                        start = time.time()
                        
                        # Handle supervised vs unsupervised training
                        if model_name.endswith('(U)') or model_type == "Unsupervised":
                            # Unsupervised training
                            from sklearn.preprocessing import StandardScaler
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X_sub)
                            model.fit(X_scaled)
                        else:
                            # Supervised training
                            model.fit(X_sub, y_sub)
                        
                        elapsed = time.time() - start
                        elapsed = max(elapsed, 0.001)  # Minimum 1ms
                        times.append(elapsed)
                        
                        current_op += 1
                        progress_bar.progress(current_op / total_operations)
                    
                    training_times[model_name] = times
                
                # Create visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                data_sizes = [int(f * len(X)) for f in fractions]
                
                # Plot 1: Training time vs data size
                for model_name, times in training_times.items():
                    ax1.plot(data_sizes, times, 'o-', label=model_name, linewidth=2)
                
                ax1.set_xlabel("Training Set Size")
                ax1.set_ylabel("Training Time (seconds)")
                ax1.set_title("Training Time vs Data Size")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Training efficiency (samples per second)
                for model_name, times in training_times.items():
                    efficiency = [size / max(time, 0.001) for size, time in zip(data_sizes, times)]
                    ax2.plot(data_sizes, efficiency, 'o-', label=model_name, linewidth=2)
                
                ax2.set_xlabel("Training Set Size")
                ax2.set_ylabel("Samples per Second")
                ax2.set_title("Training Efficiency")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Efficiency summary
                st.subheader("Training Efficiency Summary")
                efficiency_data = []
                for model_name, times in training_times.items():
                    avg_time = np.mean(times)
                    max_throughput = max([data_sizes[i] / max(times[i], 0.001) for i in range(len(times))])
                    efficiency_data.append({
                        'Model': model_name,
                        'Type': 'Supervised' if not model_name.endswith('(U)') and model_type != "Unsupervised" else 'Unsupervised',
                        'Avg Training Time (s)': f"{avg_time:.3f}",
                        'Max Throughput (samples/s)': f"{max_throughput:.1f}"
                    })
                
                import pandas as pd
                efficiency_df = pd.DataFrame(efficiency_data)
                st.dataframe(efficiency_df, use_container_width=True)

    with col2:
        st.markdown("### Efficiency Guidelines")
        st.info("""
        **Training Efficiency Factors:**
        
        **Supervised Models:**
        - Need both features and labels
        - Generally more complex training
        
        **Unsupervised Models:**
        - Only use features
        - Often faster per iteration
        - May need more iterations
        
        **Scaling Patterns:**
        - Linear: Good scalability
        - Sub-linear: Excellent efficiency
        - Super-linear: May need optimization
        """)

def _render_inference_speed_analysis():
    """Render inference speed analysis."""
    st.subheader("Inference Speed")
    
    st.markdown("""
    ### Prediction Performance Analysis
    
    Measure how quickly models can make predictions on new data,
    critical for real-time applications (both supervised and unsupervised).
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Model type selection
        model_type = st.selectbox(
            "Select model type for inference speed",
            options=["Supervised", "Unsupervised", "Both"],
            key="is_model_type"
        )
        
        if model_type == "Supervised":
            supervised_models = get_supervised_models()
            selected_models = st.multiselect(
                "Select supervised models for inference speed testing",
                options=list(supervised_models.keys()),
                default=list(supervised_models.keys())[:3],
                key="is_supervised_models"
            )
            models_dict = {name: model for name, model in supervised_models.items() if name in selected_models}
        elif model_type == "Unsupervised":
            unsupervised_models = get_unsupervised_models()
            selected_models = st.multiselect(
                "Select unsupervised models for inference speed testing",
                options=list(unsupervised_models.keys()),
                default=list(unsupervised_models.keys())[:3],
                key="is_unsupervised_models"
            )
            models_dict = {name: model for name, model in unsupervised_models.items() if name in selected_models}
        else:  # Both
            supervised_models = get_supervised_models()
            unsupervised_models = get_unsupervised_models()
            
            selected_supervised = st.multiselect(
                "Select supervised models",
                options=list(supervised_models.keys()),
                default=list(supervised_models.keys())[:2],
                key="is_both_supervised"
            )
            selected_unsupervised = st.multiselect(
                "Select unsupervised models",
                options=list(unsupervised_models.keys()),
                default=list(unsupervised_models.keys())[:2],
                key="is_both_unsupervised"
            )
            
            models_dict = {}
            models_dict.update({f"{name} (S)": model for name, model in supervised_models.items() if name in selected_supervised})
            models_dict.update({f"{name} (U)": model for name, model in unsupervised_models.items() if name in selected_unsupervised})
            selected_models = list(models_dict.keys())
        
        batch_sizes = st.multiselect(
            "Select batch sizes to test",
            options=[1, 10, 100, 1000, 5000],
            default=[1, 100, 1000],
            key="batch_sizes"
        )
        
        if st.button("Measure Inference Speed", key="is_btn"):
            if selected_models and batch_sizes:
                X, y = load_data(use_synthetic=True)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                # Preprocess data for unsupervised models
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train all models first
                trained_models = {}
                for model_name in selected_models:
                    model = clone(models_dict[model_name])
                    
                    if model_name.endswith('(U)') or model_type == "Unsupervised":
                        # Unsupervised training
                        model.fit(X_train_scaled)
                    else:
                        # Supervised training
                        model.fit(X_train, y_train)
                    
                    trained_models[model_name] = model
                
                # Test inference speeds
                results = {}
                
                progress_bar = st.progress(0)
                total_tests = len(selected_models) * len(batch_sizes)
                current_test = 0
                
                for model_name, model in trained_models.items():
                    model_results = {}
                    
                    for batch_size in batch_sizes:
                        if batch_size <= len(X_test):
                            # Choose appropriate test data
                            if model_name.endswith('(U)') or model_type == "Unsupervised":
                                X_batch = X_test_scaled[:batch_size]
                            else:
                                X_batch = X_test[:batch_size]
                            
                            # Warm up
                            if model_name.endswith('(U)') or model_type == "Unsupervised":
                                # For unsupervised models, use predict or transform
                                if hasattr(model, 'predict'):
                                    model.predict(X_batch[:min(10, batch_size)])
                                elif hasattr(model, 'transform'):
                                    model.transform(X_batch[:min(10, batch_size)])
                            else:
                                model.predict(X_batch[:min(10, batch_size)])
                            
                            # Measure inference time
                            times = []
                            for _ in range(5):  # Multiple runs for accuracy
                                start = time.time()
                                
                                if model_name.endswith('(U)') or model_type == "Unsupervised":
                                    # For unsupervised models
                                    if hasattr(model, 'predict'):
                                        predictions = model.predict(X_batch)
                                    elif hasattr(model, 'transform'):
                                        predictions = model.transform(X_batch)
                                    elif hasattr(model, 'score_samples'):
                                        predictions = model.score_samples(X_batch)
                                    else:
                                        # Fallback to decision_function if available
                                        predictions = model.decision_function(X_batch) if hasattr(model, 'decision_function') else np.zeros(batch_size)
                                else:
                                    predictions = model.predict(X_batch)
                                
                                elapsed = time.time() - start
                                elapsed = max(elapsed, 0.0001)  # Minimum 0.1ms
                                times.append(elapsed)
                            
                            avg_time = np.mean(times)
                            throughput = batch_size / avg_time
                            latency = avg_time / batch_size * 1000  # ms per sample
                            
                            model_results[batch_size] = {
                                'time': avg_time,
                                'throughput': throughput,
                                'latency': latency
                            }
                        
                        current_test += 1
                        progress_bar.progress(current_test / total_tests)
                    
                    results[model_name] = model_results
                
                # Visualize results
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Plot 1: Throughput vs batch size
                for model_name, model_results in results.items():
                    sizes = list(model_results.keys())
                    throughputs = [model_results[size]['throughput'] for size in sizes]
                    ax1.plot(sizes, throughputs, 'o-', label=model_name, linewidth=2)
                
                ax1.set_xlabel("Batch Size")
                ax1.set_ylabel("Throughput (predictions/second)")
                ax1.set_title("Inference Throughput")
                ax1.set_xscale('log')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Latency vs batch size
                for model_name, model_results in results.items():
                    sizes = list(model_results.keys())
                    latencies = [model_results[size]['latency'] for size in sizes]
                    ax2.plot(sizes, latencies, 'o-', label=model_name, linewidth=2)
                
                ax2.set_xlabel("Batch Size")
                ax2.set_ylabel("Latency (ms per prediction)")
                ax2.set_title("Prediction Latency")
                ax2.set_xscale('log')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Summary table
                st.subheader("Inference Speed Summary")
                summary_data = []
                for model_name, model_results in results.items():
                    largest_batch = max(model_results.keys())
                    stats = model_results[largest_batch]
                    model_type_label = 'Unsupervised' if model_name.endswith('(U)') or model_type == "Unsupervised" else 'Supervised'
                    summary_data.append({
                        'Model': model_name,
                        'Type': model_type_label,
                        'Max Throughput (pred/s)': f"{stats['throughput']:.1f}",
                        'Min Latency (ms)': f"{stats['latency']:.3f}",
                        'Batch Size': largest_batch
                    })
                
                import pandas as pd
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)

    with col2:
        st.markdown("### Speed Guidelines")
        st.info("""
        **Inference Speed Factors:**
        
        **Supervised Models:**
        - Direct prediction/classification
        - Usually faster single predictions
        
        **Unsupervised Models:**
        - Clustering/anomaly scoring
        - May involve distance calculations
        - Can be slower for single samples
        
        **Optimization Tips:**
        - Batch processing for efficiency
        - Model-specific optimizations
        - Hardware acceleration
        """)

def _render_memory_analysis():
    """Render memory usage analysis."""
    st.subheader("Memory Usage Analysis")
    
    st.markdown("""
    ### Model Memory Footprint
    
    Analyze memory requirements for different models (supervised and unsupervised)
    to inform deployment decisions and resource planning.
    """)
    
    # Model type selection
    model_type = st.selectbox(
        "Select model type for memory analysis",
        options=["Supervised", "Unsupervised", "Both"],
        key="ma_model_type"
    )
    
    if st.button("Analyze Memory Usage", key="ma_btn"):
        memory_results = {}
        
        X, y = load_data(use_synthetic=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Preprocess data for unsupervised models
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Get models based on selection
        if model_type == "Supervised":
            models_to_analyze = get_supervised_models()
            model_labels = {name: f"{name}" for name in models_to_analyze.keys()}
        elif model_type == "Unsupervised":
            models_to_analyze = get_unsupervised_models()
            model_labels = {name: f"{name}" for name in models_to_analyze.keys()}
        else:  # Both
            supervised_models = get_supervised_models()
            unsupervised_models = get_unsupervised_models()
            models_to_analyze = {}
            models_to_analyze.update(supervised_models)
            models_to_analyze.update(unsupervised_models)
            model_labels = {}
            model_labels.update({name: f"{name} (S)" for name in supervised_models.keys()})
            model_labels.update({name: f"{name} (U)" for name in unsupervised_models.keys()})
        
        progress_bar = st.progress(0)
        
        for i, (model_name, model) in enumerate(models_to_analyze.items()):
            try:
                # Train the model
                trained_model = clone(model)
                
                if model_type == "Unsupervised" or (model_type == "Both" and model_name in get_unsupervised_models()):
                    trained_model.fit(X_train_scaled)
                else:
                    trained_model.fit(X_train, y_train)
                
                # Measure serialized size
                serialized = pickle.dumps(trained_model)
                model_size = len(serialized)
                
                # Estimate parameter count (simplified)
                param_count = 0
                if hasattr(trained_model, 'coef_'):
                    if trained_model.coef_.ndim == 1:
                        param_count = len(trained_model.coef_)
                    else:
                        param_count = trained_model.coef_.size
                elif hasattr(trained_model, 'feature_importances_'):
                    param_count = len(trained_model.feature_importances_)
                elif hasattr(trained_model, 'cluster_centers_'):
                    param_count = trained_model.cluster_centers_.size
                elif hasattr(trained_model, 'components_'):
                    param_count = trained_model.components_.size
                elif hasattr(trained_model, 'n_features_in_'):
                    param_count = trained_model.n_features_in_
                else:
                    param_count = X_train.shape[1]  # Fallback to feature count
                
                display_name = model_labels.get(model_name, model_name)
                memory_results[display_name] = {
                    'serialized_size_kb': model_size / 1024,
                    'parameter_count': param_count,
                    'size_per_param': model_size / max(param_count, 1),
                    'model_type': 'Unsupervised' if model_type == "Unsupervised" or (model_type == "Both" and model_name in get_unsupervised_models()) else 'Supervised'
                }
                
            except Exception as e:
                st.warning(f"Could not analyze {model_name}: {str(e)}")
                display_name = model_labels.get(model_name, model_name)
                memory_results[display_name] = {
                    'serialized_size_kb': 0,
                    'parameter_count': 0,
                    'size_per_param': 0,
                    'model_type': 'Unknown'
                }
            
            progress_bar.progress((i + 1) / len(models_to_analyze))
        
        # Visualize results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        models = list(memory_results.keys())
        sizes = [memory_results[m]['serialized_size_kb'] for m in models]
        params = [memory_results[m]['parameter_count'] for m in models]
        types = [memory_results[m]['model_type'] for m in models]
        
        # Color coding for model types
        colors = ['blue' if t == 'Supervised' else 'red' if t == 'Unsupervised' else 'gray' for t in types]
        
        # Plot 1: Model sizes
        bars1 = ax1.bar(models, sizes, alpha=0.7, color=colors)
        ax1.set_ylabel("Model Size (KB)")
        ax1.set_title("Serialized Model Sizes")
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, size in zip(bars1, sizes):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.01,
                    f'{size:.1f}', ha='center', va='bottom')
        
        # Plot 2: Parameter counts
        bars2 = ax2.bar(models, params, alpha=0.7, color=colors)
        ax2.set_ylabel("Parameter Count")
        ax2.set_title("Model Parameter Counts")
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, param in zip(bars2, params):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(params)*0.01,
                    f'{param}', ha='center', va='bottom')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Supervised'),
                          Patch(facecolor='red', alpha=0.7, label='Unsupervised')]
        ax1.legend(handles=legend_elements)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Summary table
        st.subheader("Memory Usage Summary")
        import pandas as pd
        memory_df = pd.DataFrame(memory_results).T
        memory_df.index.name = 'Model'
        memory_df = memory_df.round(2)
        memory_df.columns = ['Size (KB)', 'Parameters', 'Bytes per Parameter', 'Type']
        
        # Style the dataframe
        styled_df = memory_df.style.background_gradient(subset=['Size (KB)'], cmap='Reds')
        st.dataframe(styled_df, use_container_width=True)
        
        # Memory efficiency insights
        supervised_models_df = memory_df[memory_df['Type'] == 'Supervised']
        unsupervised_models_df = memory_df[memory_df['Type'] == 'Unsupervised']
        
        st.subheader("Memory Insights")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if not memory_df['Bytes per Parameter'].empty:
                most_efficient = memory_df['Bytes per Parameter'].idxmin()
                st.metric("Most Memory Efficient", most_efficient)
        
        with col2:
            largest_model = memory_df['Size (KB)'].idxmax()
            st.metric("Largest Model", largest_model)
        
        with col3:
            if not supervised_models_df.empty:
                avg_supervised = supervised_models_df['Size (KB)'].mean()
                st.metric("Avg Supervised Size", f"{avg_supervised:.1f} KB")
            else:
                st.metric("Avg Supervised Size", "N/A")
        
        with col4:
            if not unsupervised_models_df.empty:
                avg_unsupervised = unsupervised_models_df['Size (KB)'].mean()
                st.metric("Avg Unsupervised Size", f"{avg_unsupervised:.1f} KB")
            else:
                st.metric("Avg Unsupervised Size", "N/A")

def _render_scalability_testing():
    """Render scalability testing."""
    st.subheader("Scalability Testing")
    
    st.markdown("""
    ### Model Scalability Assessment
    
    Evaluate how well models (supervised and unsupervised) scale with 
    increasing data size and complexity.
    """)
    
    st.info("""
    **Scalability Testing Components:**
    
    📈 **Data Size Scaling:** How performance changes with dataset size
    🔢 **Feature Scaling:** Impact of increasing feature dimensions  
    👥 **Concurrent Users:** Multi-user performance simulation
    🖥️ **Hardware Scaling:** Performance across different hardware configurations
    🤖 **Model Type Comparison:** Supervised vs Unsupervised scaling patterns
    
    **Note:** Full scalability testing requires controlled infrastructure environments.
    This section provides framework for scalability analysis.
    """)
    
    # Model type for scalability simulation
    model_type = st.selectbox(
        "Select model type for scalability simulation",
        options=["Supervised", "Unsupervised", "Comparison"],
        key="scale_model_type"
    )
    
    # Simplified scalability simulation
    if st.button("Simulate Scalability Analysis", key="scale_btn"):
        # Simulate performance metrics at different scales
        data_sizes = np.array([1000, 5000, 10000, 25000, 50000])
        
        if model_type == "Supervised":
            # Simulate supervised model scaling patterns
            linear_perf = 1 - (data_sizes - 1000) / 50000 * 0.2  # Slight degradation
            sublinear_perf = 1 - np.sqrt((data_sizes - 1000) / 50000) * 0.3  # Better scaling
            superlinear_perf = 1 - ((data_sizes - 1000) / 50000) ** 1.5 * 0.4  # Poor scaling
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Performance vs data size
            ax1.plot(data_sizes, linear_perf, 'o-', label='Linear Scaling Model', linewidth=2)
            ax1.plot(data_sizes, sublinear_perf, 'o-', label='Sub-linear Scaling Model', linewidth=2)
            ax1.plot(data_sizes, superlinear_perf, 'o-', label='Super-linear Scaling Model', linewidth=2)
            
            ax1.set_xlabel('Data Size')
            ax1.set_ylabel('Performance Score')
            ax1.set_title('Supervised Model Performance Scaling')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Training time scaling
            training_times = {
                'Linear': data_sizes * 0.001,
                'Sub-linear': data_sizes ** 0.8 * 0.0005,
                'Super-linear': data_sizes ** 1.3 * 0.0002
            }
            
            for pattern, times in training_times.items():
                ax2.plot(data_sizes, times, 'o-', label=f'{pattern} Scaling', linewidth=2)
            
            ax2.set_xlabel('Data Size')
            ax2.set_ylabel('Training Time (relative)')
            ax2.set_title('Supervised Training Time Scaling')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
        elif model_type == "Unsupervised":
            # Simulate unsupervised model scaling patterns
            clustering_quality = 0.8 - (data_sizes - 1000) / 50000 * 0.1  # Slight degradation
            convergence_speed = 1 / (1 + np.exp(-(data_sizes - 25000) / 10000))  # Sigmoid
            memory_usage = data_sizes ** 1.2 / 1000  # Super-linear memory growth
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Clustering quality vs data size
            ax1.plot(data_sizes, clustering_quality, 'o-', label='Clustering Quality', linewidth=2, color='blue')
            ax1_twin = ax1.twinx()
            ax1_twin.plot(data_sizes, convergence_speed, 'o-', label='Convergence Speed', linewidth=2, color='red')
            
            ax1.set_xlabel('Data Size')
            ax1.set_ylabel('Clustering Quality', color='blue')
            ax1_twin.set_ylabel('Convergence Speed', color='red')
            ax1.set_title('Unsupervised Model Quality Scaling')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Memory usage scaling
            ax2.plot(data_sizes, memory_usage, 'o-', label='Memory Usage', linewidth=2, color='green')
            ax2.set_xlabel('Data Size')
            ax2.set_ylabel('Memory Usage (relative)')
            ax2.set_title('Unsupervised Memory Scaling')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
        else:  # Comparison
            # Compare supervised vs unsupervised scaling
            sup_perf = 1 - (data_sizes - 1000) / 50000 * 0.2
            unsup_perf = 0.8 - (data_sizes - 1000) / 50000 * 0.1
            
            sup_time = data_sizes * 0.001
            unsup_time = data_sizes ** 0.9 * 0.0008  # Slightly better scaling
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Performance comparison
            ax1.plot(data_sizes, sup_perf, 'o-', label='Supervised Models', linewidth=2, color='blue')
            ax1.plot(data_sizes, unsup_perf, 'o-', label='Unsupervised Models', linewidth=2, color='red')
            
            ax1.set_xlabel('Data Size')
            ax1.set_ylabel('Performance Score')
            ax1.set_title('Performance Scaling Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Training time comparison
            ax2.plot(data_sizes, sup_time, 'o-', label='Supervised Training', linewidth=2, color='blue')
            ax2.plot(data_sizes, unsup_time, 'o-', label='Unsupervised Training', linewidth=2, color='red')
            
            ax2.set_xlabel('Data Size')
            ax2.set_ylabel('Training Time (relative)')
            ax2.set_title('Training Time Scaling Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.success("Scalability simulation complete! In production, test with actual hardware and data.")

def _render_resource_optimization():
    """Render resource optimization analysis."""
    st.subheader("Resource Optimization")
    
    st.markdown("""
    ### Optimization Strategies & Recommendations
    
    Guidelines for optimizing model performance and resource utilization
    in production environments.
    """)
    
    # Optimization recommendations
    st.subheader("🎯 Optimization Strategies")
    
    optimization_tabs = st.tabs([
        "Model Selection",
        "Hardware Optimization", 
        "Software Optimization",
        "Deployment Strategies"
    ])
    
    with optimization_tabs[0]:
        st.markdown("""
        **Model Selection for Efficiency:**
        
        🚀 **High-Speed Requirements:**
        - Linear models (Logistic Regression, SVM)
        - Tree-based models with depth limits
        - Avoid complex ensemble methods
        
        🎯 **Balanced Performance:**
        - Random Forest with optimized parameters
        - Gradient Boosting with early stopping
        - Neural networks with pruning
        
        💾 **Memory-Constrained Environments:**
        - Linear models with regularization
        - Decision trees with max depth
        - Avoid large ensemble methods
        """)
    
    with optimization_tabs[1]:
        st.markdown("""
        **Hardware Optimization:**
        
        🖥️ **CPU Optimization:**
        - Use multiple cores for tree-based models
        - Optimize BLAS libraries for linear algebra
        - Consider CPU-specific optimizations
        
        🚀 **GPU Acceleration:**
        - Use GPU-enabled frameworks for neural networks
        - Consider GPU-accelerated XGBoost/LightGBM
        - Batch processing for inference
        
        💿 **Memory Management:**
        - Use memory mapping for large datasets
        - Implement data streaming for training
        - Consider distributed computing for very large data
        """)
    
    with optimization_tabs[2]:
        st.markdown("""
        **Software Optimization:**
        
        ⚡ **Code Optimization:**
        - Use compiled libraries (NumPy, Scikit-learn)
        - Implement model caching
        - Optimize data preprocessing pipelines
        
        📦 **Model Compression:**
        - Quantization for neural networks
        - Feature selection to reduce model size
        - Model distillation techniques
        
        🔄 **Pipeline Optimization:**
        - Parallel feature computation
        - Batch prediction processing
        - Asynchronous model serving
        """)
    
    with optimization_tabs[3]:
        st.markdown("""
        **Deployment Strategies:**
        
        ☁️ **Cloud Deployment:**
        - Auto-scaling based on load
        - Load balancing across instances
        - Use managed ML services when appropriate
        
        🏠 **Edge Deployment:**
        - Model quantization and pruning
        - Optimize for specific hardware
        - Consider federated learning approaches
        
        📊 **Monitoring & Optimization:**
        - Real-time performance monitoring
        - A/B testing for model versions
        - Continuous performance optimization
        """)
    
    # Resource usage calculator
    st.subheader("📊 Resource Usage Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        predictions_per_day = st.number_input("Predictions per day", min_value=1, max_value=10000000, value=10000)
        avg_latency_ms = st.number_input("Average latency (ms)", min_value=1.0, max_value=10000.0, value=50.0)
        model_size_mb = st.number_input("Model size (MB)", min_value=0.1, max_value=1000.0, value=5.0)
    
    with col2:
        # Calculate resource requirements
        predictions_per_second = predictions_per_day / (24 * 3600)
        cpu_utilization = min(100, predictions_per_second * avg_latency_ms / 1000 * 100)
        memory_requirement = model_size_mb * 2  # Assume 2x overhead
        
        st.metric("Predictions per second", f"{predictions_per_second:.2f}")
        st.metric("Estimated CPU utilization", f"{cpu_utilization:.1f}%")
        st.metric("Estimated memory requirement", f"{memory_requirement:.1f} MB")
        
        # Recommendations based on calculations
        if cpu_utilization > 80:
            st.warning("⚠️ High CPU utilization expected. Consider optimization or scaling.")
        elif cpu_utilization < 10:
            st.info("💡 Low resource usage. Single instance should suffice.")
        else:
            st.success("✅ Reasonable resource requirements.")
