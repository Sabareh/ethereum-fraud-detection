import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render_data_management_page():
    """Render the data management page."""
    st.title("🗄️ Data Management & Pipeline")
    
    st.markdown("""
    # Ethereum Fraud Detection Data Pipeline
    
    This page provides comprehensive tools for managing the data pipeline used in fraud detection,
    including data ingestion, preprocessing, quality monitoring, and dataset management.
    """)
    
    # Create tabs for different data management functions
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview",
        "🔄 Data Pipeline",
        "✅ Quality Control",
        "📥 Data Sources",
        "⚙️ Configuration"
    ])
    
    with tab1:
        render_data_overview_tab()
    
    with tab2:
        render_data_pipeline_tab()
    
    with tab3:
        render_quality_control_tab()
    
    with tab4:
        render_data_sources_tab()
    
    with tab5:
        render_configuration_tab()

def render_data_overview_tab():
    """Render data overview and statistics."""
    st.header("📊 Dataset Overview")
    
    st.markdown("""
    ### Current Dataset Statistics
    
    Monitor the current state of the fraud detection dataset including size,
    distribution, and key characteristics.
    """)
    
    # Dataset summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Addresses", "9,841", "+127 today")
    with col2:
        st.metric("Fraudulent", "2,179", "+23 today")
    with col3:
        st.metric("Legitimate", "7,662", "+104 today")
    with col4:
        st.metric("Fraud Rate", "22.1%", "-0.3%")
    
    # Dataset composition visualization
    st.subheader("📈 Dataset Composition Over Time")
    
    # Generate sample data for visualization
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='W')
    legitimate_cumulative = np.cumsum(np.random.poisson(25, len(dates)))
    fraudulent_cumulative = np.cumsum(np.random.poisson(8, len(dates)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=legitimate_cumulative,
        mode='lines', name='Legitimate Addresses',
        line=dict(color='green', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=fraudulent_cumulative,
        mode='lines', name='Fraudulent Addresses',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title='Cumulative Address Count Over Time',
        xaxis_title='Date',
        yaxis_title='Number of Addresses',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature statistics
    st.subheader("🔢 Feature Statistics")
    
    # Mock feature statistics
    feature_stats = pd.DataFrame({
        'Feature': [
            'Avg min between sent tnx',
            'Avg min between received tnx', 
            'Time Diff between first and last',
            'Sent tnx',
            'Received Tnx',
            'Number of Created Contracts',
            'Unique Received From Addresses',
            'Unique Sent To Addresses',
            'Min value received',
            'Max value received',
            'Avg val received',
            'Avg val sent',
            'Total Ether sent',
            'Total ether received',
            'Total ether balance',
            'Total ERC20 tnxs',
            'ERC20 total Ether received',
            'ERC20 total ether sent',
            'ERC20 uniq sent addr',
            'ERC20 uniq rec addr'
        ],
        'Mean': np.random.uniform(0.1, 100, 20),
        'Std': np.random.uniform(0.05, 50, 20),
        'Min': np.random.uniform(0, 1, 20),
        'Max': np.random.uniform(500, 10000, 20),
        'Missing_Count': np.random.randint(0, 100, 20),
        'Missing_Percentage': np.random.uniform(0, 10, 20)
    })
    
    # Format the dataframe
    feature_stats_display = feature_stats.copy()
    for col in ['Mean', 'Std', 'Min', 'Max']:
        feature_stats_display[col] = feature_stats_display[col].apply(lambda x: f"{x:.2f}")
    feature_stats_display['Missing_Percentage'] = feature_stats_display['Missing_Percentage'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(feature_stats_display, use_container_width=True)
    
    # Data quality indicators
    st.subheader("🎯 Data Quality Indicators")
    
    quality_col1, quality_col2, quality_col3 = st.columns(3)
    
    with quality_col1:
        st.metric("Completeness", "94.2%", "↑1.2%")
        st.caption("Percentage of non-missing values")
    
    with quality_col2:
        st.metric("Consistency", "97.8%", "↑0.5%")
        st.caption("Data format consistency")
    
    with quality_col3:
        st.metric("Accuracy", "96.1%", "↓0.3%")
        st.caption("Estimated data accuracy")

def render_data_pipeline_tab():
    """Render data pipeline management interface."""
    st.header("🔄 Data Processing Pipeline")
    
    st.markdown("""
    ### Pipeline Status & Management
    
    Monitor and control the data processing pipeline from raw blockchain data
    to machine learning ready features.
    """)
    
    # Pipeline status
    st.subheader("🚦 Pipeline Status")
    
    pipeline_stages = [
        {"stage": "Data Extraction", "status": "Running", "last_run": "2 min ago", "success_rate": "99.2%"},
        {"stage": "Data Validation", "status": "Completed", "last_run": "5 min ago", "success_rate": "97.8%"},
        {"stage": "Feature Engineering", "status": "Running", "last_run": "1 min ago", "success_rate": "98.5%"},
        {"stage": "Data Transformation", "status": "Pending", "last_run": "15 min ago", "success_rate": "96.9%"},
        {"stage": "Quality Checks", "status": "Completed", "last_run": "8 min ago", "success_rate": "95.7%"},
        {"stage": "Model Training", "status": "Idle", "last_run": "2 hours ago", "success_rate": "94.3%"}
    ]
    
    for stage in pipeline_stages:
        col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
        
        with col1:
            status_color = {
                "Running": "🟢",
                "Completed": "🟦", 
                "Pending": "🟡",
                "Idle": "⚪",
                "Error": "🔴"
            }
            st.write(f"{status_color[stage['status']]} **{stage['stage']}**")
        
        with col2:
            st.write(f"Status: {stage['status']}")
        
        with col3:
            st.write(f"Last run: {stage['last_run']}")
        
        with col4:
            st.write(f"Success: {stage['success_rate']}")
    
    # Pipeline controls
    st.subheader("⚙️ Pipeline Controls")
    
    control_col1, control_col2, control_col3, control_col4 = st.columns(4)
    
    with control_col1:
        if st.button("🚀 Start Full Pipeline", type="primary"):
            st.success("Pipeline started successfully!")
    
    with control_col2:
        if st.button("⏸️ Pause Pipeline"):
            st.warning("Pipeline paused.")
    
    with control_col3:
        if st.button("🔄 Restart Failed"):
            st.info("Restarting failed stages...")
    
    with control_col4:
        if st.button("📊 View Logs"):
            st.info("Opening pipeline logs...")
    
    # Processing timeline
    st.subheader("📅 Processing Timeline")
    
    # Generate sample timeline data
    timeline_dates = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                                 end=datetime.now(), freq='H')
    processed_records = np.random.poisson(150, len(timeline_dates))
    
    fig = px.bar(x=timeline_dates, y=processed_records,
                title="Records Processed Over Last 24 Hours",
                labels={'x': 'Time', 'y': 'Records Processed'})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Resource utilization
    st.subheader("💻 Resource Utilization")
    
    resource_col1, resource_col2, resource_col3, resource_col4 = st.columns(4)
    
    with resource_col1:
        st.metric("CPU Usage", "67%", "↑5%")
    with resource_col2:
        st.metric("Memory Usage", "4.2 GB", "↑0.3 GB")
    with resource_col3:
        st.metric("Disk I/O", "245 MB/s", "↓12 MB/s")
    with resource_col4:
        st.metric("Network", "89 MB/s", "↑15 MB/s")

def render_quality_control_tab():
    """Render data quality control interface."""
    st.header("✅ Data Quality Control")
    
    st.markdown("""
    ### Automated Quality Monitoring
    
    Continuous monitoring of data quality with automated alerts for anomalies,
    missing values, and data drift detection.
    """)
    
    # Quality score dashboard
    st.subheader("📊 Quality Score Dashboard")
    
    # Overall quality score gauge
    overall_quality = 94.2
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = overall_quality,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Overall Data Quality Score"},
        delta = {'reference': 95},
        gauge = {'axis': {'range': [None, 100]},
                 'bar': {'color': "darkblue"},
                 'steps': [
                     {'range': [0, 70], 'color': "lightgray"},
                     {'range': [70, 85], 'color': "yellow"},
                     {'range': [85, 95], 'color': "lightgreen"},
                     {'range': [95, 100], 'color': "green"}],
                 'threshold': {'line': {'color': "red", 'width': 4},
                              'thickness': 0.75,
                              'value': 90}}))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Quality dimensions
    st.subheader("🎯 Quality Dimensions")
    
    quality_dimensions = pd.DataFrame({
        'Dimension': ['Completeness', 'Accuracy', 'Consistency', 'Timeliness', 'Validity'],
        'Score': [94.2, 96.1, 97.8, 92.3, 95.7],
        'Threshold': [90, 95, 95, 85, 90],
        'Status': ['Pass', 'Pass', 'Pass', 'Pass', 'Pass'],
        'Issues': [23, 12, 8, 45, 18]
    })
    
    # Color code based on pass/fail
    def color_status(val):
        return 'background-color: lightgreen' if val == 'Pass' else 'background-color: lightcoral'
    
    styled_quality = quality_dimensions.style.applymap(color_status, subset=['Status'])
    st.dataframe(styled_quality, use_container_width=True)
    
    # Data drift detection
    st.subheader("📈 Data Drift Detection")
    
    st.markdown("""
    Monitor statistical properties of incoming data to detect distribution shifts
    that might affect model performance.
    """)
    
    # Generate sample drift data
    feature_names = ['Transaction_Frequency', 'Account_Age', 'Value_Patterns', 'Network_Activity']
    drift_scores = np.random.uniform(0.05, 0.25, len(feature_names))
    drift_threshold = 0.15
    
    fig = go.Figure()
    colors = ['red' if score > drift_threshold else 'green' for score in drift_scores]
    
    fig.add_trace(go.Bar(
        x=feature_names,
        y=drift_scores,
        marker_color=colors,
        name='Drift Score'
    ))
    
    fig.add_hline(y=drift_threshold, line_dash="dash", line_color="red", 
                  annotation_text="Drift Threshold")
    
    fig.update_layout(
        title='Feature Drift Scores',
        xaxis_title='Features',
        yaxis_title='Drift Score',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Alert summary
    st.subheader("🚨 Quality Alerts")
    
    alerts = [
        {"time": "10 min ago", "type": "Warning", "message": "Missing values increased in 'Avg min between sent tnx'"},
        {"time": "1 hour ago", "type": "Info", "message": "Data validation completed successfully"},
        {"time": "3 hours ago", "type": "Error", "message": "Feature drift detected in Transaction_Frequency"},
        {"time": "6 hours ago", "type": "Info", "message": "Quality check passed for all dimensions"}
    ]
    
    for alert in alerts:
        alert_color = {
            "Error": "🔴",
            "Warning": "🟡", 
            "Info": "🔵"
        }
        st.write(f"{alert_color[alert['type']]} **{alert['time']}** - {alert['message']}")

def render_data_sources_tab():
    """Render data sources management interface."""
    st.header("📥 Data Sources Management")
    
    st.markdown("""
    ### Configure and Monitor Data Sources
    
    Manage connections to various data sources including blockchain nodes,
    external APIs, and labeled datasets.
    """)
    
    # Data source status
    st.subheader("🔗 Data Source Status")
    
    data_sources = [
        {
            "name": "Ethereum Mainnet Node",
            "type": "Blockchain",
            "status": "Connected",
            "last_sync": "30 sec ago",
            "records_today": "1,247",
            "uptime": "99.8%"
        },
        {
            "name": "Etherscan API",
            "type": "External API", 
            "status": "Connected",
            "last_sync": "2 min ago",
            "records_today": "892",
            "uptime": "97.2%"
        },
        {
            "name": "CipherTrace Labels",
            "type": "Label Provider",
            "status": "Connected",
            "last_sync": "15 min ago", 
            "records_today": "156",
            "uptime": "99.1%"
        },
        {
            "name": "Elliptic Dataset",
            "type": "Static Dataset",
            "status": "Loaded",
            "last_sync": "Daily",
            "records_today": "0",
            "uptime": "100%"
        },
        {
            "name": "DeFi Pulse Data",
            "type": "External API",
            "status": "Error",
            "last_sync": "2 hours ago",
            "records_today": "0", 
            "uptime": "85.3%"
        }
    ]
    
    for source in data_sources:
        with st.expander(f"🔌 {source['name']} - {source['status']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Type:** {source['type']}")
                st.write(f"**Status:** {source['status']}")
                st.write(f"**Last Sync:** {source['last_sync']}")
            
            with col2:
                st.write(f"**Records Today:** {source['records_today']}")
                st.write(f"**Uptime:** {source['uptime']}")
                
                if source['status'] == 'Error':
                    if st.button(f"🔄 Retry Connection", key=f"retry_{source['name']}"):
                        st.success("Attempting to reconnect...")
    
    # Add new data source
    st.subheader("➕ Add New Data Source")
    
    with st.form("add_data_source"):
        col1, col2 = st.columns(2)
        
        with col1:
            source_name = st.text_input("Data Source Name")
            source_type = st.selectbox("Source Type", 
                                     ["Blockchain Node", "External API", "Database", "File Upload"])
        
        with col2:
            connection_string = st.text_input("Connection String/URL")
            update_frequency = st.selectbox("Update Frequency",
                                          ["Real-time", "Every minute", "Hourly", "Daily"])
        
        authentication = st.text_area("Authentication Details (JSON format)")
        
        if st.form_submit_button("🔗 Add Data Source"):
            st.success(f"Data source '{source_name}' added successfully!")
    
    # Data source analytics
    st.subheader("📊 Source Analytics")
    
    # Generate sample analytics data
    source_names = [source['name'] for source in data_sources]
    daily_records = [int(source['records_today'].replace(',', '')) for source in data_sources]
    
    fig = px.pie(values=daily_records, names=source_names,
                title="Records by Data Source (Today)")
    st.plotly_chart(fig, use_container_width=True)

def render_configuration_tab():
    """Render configuration management interface."""
    st.header("⚙️ Configuration Management")
    
    st.markdown("""
    ### System Configuration
    
    Manage system-wide settings, feature engineering parameters,
    and model training configurations.
    """)
    
    # Configuration sections
    config_section = st.selectbox("Configuration Section", [
        "Data Processing",
        "Feature Engineering", 
        "Model Training",
        "Quality Thresholds",
        "System Settings"
    ])
    
    if config_section == "Data Processing":
        st.subheader("📊 Data Processing Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            batch_size = st.number_input("Batch Size", value=1000, min_value=100)
            processing_threads = st.slider("Processing Threads", 1, 8, 4)
            data_retention_days = st.number_input("Data Retention (days)", value=365)
        
        with col2:
            enable_caching = st.checkbox("Enable Caching", value=True)
            auto_backup = st.checkbox("Auto Backup", value=True)
            compression_enabled = st.checkbox("Enable Compression", value=True)
    
    elif config_section == "Feature Engineering":
        st.subheader("🔧 Feature Engineering Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            time_window_hours = st.number_input("Time Window (hours)", value=24)
            min_transaction_count = st.number_input("Min Transaction Count", value=5)
            outlier_threshold = st.slider("Outlier Threshold (std)", 1.0, 5.0, 3.0)
        
        with col2:
            normalize_features = st.checkbox("Normalize Features", value=True)
            log_transform = st.checkbox("Apply Log Transform", value=True)
            handle_missing = st.selectbox("Missing Value Strategy", 
                                        ["Mean", "Median", "Mode", "Forward Fill"])
    
    elif config_section == "Model Training":
        st.subheader("🤖 Model Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            train_test_split = st.slider("Train/Test Split", 0.1, 0.9, 0.7)
            cross_validation_folds = st.number_input("CV Folds", value=5)
            random_seed = st.number_input("Random Seed", value=42)
        
        with col2:
            auto_retrain = st.checkbox("Auto Retrain", value=False)
            retrain_threshold = st.slider("Retrain Threshold (drift)", 0.05, 0.5, 0.15)
            model_versioning = st.checkbox("Enable Model Versioning", value=True)
    
    elif config_section == "Quality Thresholds":
        st.subheader("✅ Quality Threshold Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            completeness_threshold = st.slider("Completeness (%)", 70, 100, 90)
            accuracy_threshold = st.slider("Accuracy (%)", 80, 100, 95)
            consistency_threshold = st.slider("Consistency (%)", 80, 100, 95)
        
        with col2:
            drift_threshold = st.slider("Drift Threshold", 0.05, 0.5, 0.15)
            alert_sensitivity = st.selectbox("Alert Sensitivity", ["Low", "Medium", "High"])
            notification_email = st.text_input("Notification Email")
    
    else:  # System Settings
        st.subheader("🖥️ System Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"])
            max_memory_gb = st.number_input("Max Memory (GB)", value=8)
            api_rate_limit = st.number_input("API Rate Limit", value=1000)
        
        with col2:
            enable_monitoring = st.checkbox("Enable Monitoring", value=True)
            debug_mode = st.checkbox("Debug Mode", value=False)
            performance_tracking = st.checkbox("Performance Tracking", value=True)
    
    # Save configuration
    if st.button("💾 Save Configuration", type="primary"):
        st.success("Configuration saved successfully!")
    
    # Configuration history
    st.subheader("📝 Configuration History")
    
    config_history = pd.DataFrame({
        'Timestamp': [
            '2024-01-15 10:30:00',
            '2024-01-14 15:45:00',
            '2024-01-13 09:15:00',
            '2024-01-12 14:20:00'
        ],
        'Section': ['Data Processing', 'Model Training', 'Quality Thresholds', 'System Settings'],
        'Changed_By': ['admin', 'data_scientist', 'admin', 'system'],
        'Description': [
            'Updated batch size to 1000',
            'Changed CV folds to 5',
            'Lowered drift threshold to 0.15',
            'Enabled performance tracking'
        ]
    })
    
    st.dataframe(config_history, use_container_width=True)
    
    # Export/Import configuration
    st.subheader("📤 Export/Import Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Export Config"):
            # Mock configuration export
            config_data = {
                "data_processing": {"batch_size": 1000, "threads": 4},
                "feature_engineering": {"time_window": 24, "normalize": True},
                "model_training": {"train_split": 0.7, "cv_folds": 5}
            }
            st.download_button(
                "Download Configuration",
                json.dumps(config_data, indent=2),
                "fraud_detection_config.json",
                "application/json"
            )
    
    with col2:
        uploaded_config = st.file_uploader("📥 Import Config", type=['json'])
        if uploaded_config:
            st.success("Configuration imported successfully!")
