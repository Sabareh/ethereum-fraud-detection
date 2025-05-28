import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def render_realtime_prediction_page():
    """Render the real-time prediction page."""
    st.title("Real-Time Fraud Detection")
    
    st.markdown("""
    # Live Ethereum Address Analysis
    
    This page simulates real-time fraud detection for Ethereum addresses.
    Enter address details or upload transaction data to get instant fraud risk assessments.
    """)
    
    # Create tabs for different real-time features
    tab1, tab2, tab3, tab4 = st.tabs([
        "Single Address Analysis", 
        "Batch Processing", 
        "Live Monitoring",
        "Alert Dashboard"
    ])
    
    with tab1:
        _render_single_address_tab()
    
    with tab2:
        _render_batch_processing_tab()
    
    with tab3:
        _render_live_monitoring_tab()
    
    with tab4:
        _render_alert_dashboard_tab()

def _render_single_address_tab():
    """Render single address analysis interface."""
    st.header("Single Address Risk Assessment")
    
    st.markdown("""
    ### Analyze Individual Ethereum Address
    
    Enter transaction characteristics for a specific address to get instant fraud risk assessment.
    """)
    
    # Input form for address features
    with st.form("address_analysis_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Transaction Info")
            address = st.text_input("Ethereum Address", placeholder="0x123...abc")
            sent_transactions = st.number_input("Sent Transactions", min_value=0, value=10)
            received_transactions = st.number_input("Received Transactions", min_value=0, value=15)
            total_ether_sent = st.number_input("Total Ether Sent", min_value=0.0, value=5.5, format="%.4f")
            total_ether_received = st.number_input("Total Ether Received", min_value=0.0, value=4.2, format="%.4f")
        
        with col2:
            st.subheader("Advanced Features")
            account_age_days = st.number_input("Account Age (days)", min_value=0, value=120)
            unique_recipients = st.number_input("Unique Recipients", min_value=0, value=8)
            contract_transactions = st.number_input("Contract Interactions", min_value=0, value=2)
            avg_time_between_tx = st.number_input("Avg Time Between Tx (hours)", min_value=0.0, value=24.0)
            gas_price_avg = st.number_input("Average Gas Price (Gwei)", min_value=0.0, value=20.0)
        
        submit_button = st.form_submit_button("🔍 Analyze Address", type="primary")
    
    if submit_button:
        # Create feature vector
        features = np.array([
            sent_transactions, received_transactions, total_ether_sent, total_ether_received,
            account_age_days, unique_recipients, contract_transactions, avg_time_between_tx,
            gas_price_avg, sent_transactions + received_transactions  # total transactions
        ]).reshape(1, -1)
        
        # Simulate prediction
        fraud_probability, risk_factors = _simulate_fraud_prediction(features[0])
        
        # Display results
        _display_fraud_analysis_results(address, fraud_probability, risk_factors, features[0])

def _render_batch_processing_tab():
    """Render batch processing interface."""
    st.header("Batch Address Processing")
    
    st.markdown("""
    ### Analyze Multiple Addresses Simultaneously
    
    Upload a CSV file with multiple addresses or paste address data for bulk analysis.
    """)
    
    # Batch upload option
    uploaded_file = st.file_uploader(
        "Upload CSV with address data",
        type=['csv'],
        help="CSV should contain columns for transaction features"
    )
    
    if uploaded_file:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(batch_df)} addresses for analysis")
            
            if st.button("🚀 Process Batch", type="primary"):
                progress_bar = st.progress(0)
                results_container = st.empty()
                
                # Simulate batch processing
                batch_results = []
                for i, row in batch_df.iterrows():
                    # Simulate processing time
                    time.sleep(0.1)
                    progress_bar.progress((i + 1) / len(batch_df))
                    
                    # Extract features (simplified)
                    features = row.select_dtypes(include=[np.number]).values[:10]
                    if len(features) < 10:
                        features = np.pad(features, (0, 10 - len(features)), 'constant')
                    
                    fraud_prob, _ = _simulate_fraud_prediction(features)
                    
                    batch_results.append({
                        'Address': f"0x{''.join(np.random.choice(list('0123456789abcdef'), size=8))}...",
                        'Fraud_Probability': fraud_prob,
                        'Risk_Level': _get_risk_level(fraud_prob),
                        'Processing_Time': datetime.now().strftime('%H:%M:%S')
                    })
                
                # Display results
                results_df = pd.DataFrame(batch_results)
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Processed", len(results_df))
                with col2:
                    high_risk = len(results_df[results_df['Risk_Level'] == 'High Risk'])
                    st.metric("High Risk", high_risk)
                with col3:
                    medium_risk = len(results_df[results_df['Risk_Level'] == 'Medium Risk'])
                    st.metric("Medium Risk", medium_risk)
                with col4:
                    avg_prob = results_df['Fraud_Probability'].mean()
                    st.metric("Avg Risk Score", f"{avg_prob:.2%}")
                
                # Results table
                st.subheader("Batch Processing Results")
                styled_results = results_df.style.format({
                    'Fraud_Probability': '{:.2%}'
                }).background_gradient(subset=['Fraud_Probability'], cmap='Reds')
                
                st.dataframe(styled_results, use_container_width=True)
                
                # Download results
                csv_results = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv_results,
                    "fraud_analysis_results.csv",
                    "text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Manual batch input
    st.subheader("Manual Batch Input")
    st.markdown("Paste comma-separated address data:")
    
    batch_text = st.text_area(
        "Address Data",
        placeholder="Address1,10,15,5.5,4.2,120,8,2,24.0,20.0\nAddress2,5,8,2.1,1.8,45,3,0,12.0,25.0",
        height=100
    )
    
    if batch_text and st.button("Process Text Input"):
        try:
            lines = batch_text.strip().split('\n')
            manual_results = []
            
            for i, line in enumerate(lines):
                if line.strip():
                    parts = line.split(',')
                    address = parts[0] if parts else f'Address_{i}'
                    features = np.array([float(x) if x.replace('.', '').isdigit() else 0 for x in parts[1:10]])
                    if len(features) < 9:
                        features = np.pad(features, (0, 9 - len(features)), 'constant')
                    
                    fraud_prob, _ = _simulate_fraud_prediction(features)
                    manual_results.append({
                        'Address': address,
                        'Fraud_Probability': fraud_prob,
                        'Risk_Level': _get_risk_level(fraud_prob)
                    })
            
            if manual_results:
                results_df = pd.DataFrame(manual_results)
                st.dataframe(results_df)
        except Exception as e:
            st.error(f"Error processing text input: {str(e)}")

def _render_live_monitoring_tab():
    """Render live monitoring interface."""
    st.header("Live Transaction Monitoring")
    
    st.markdown("""
    ### Real-Time Transaction Stream Simulation
    
    This simulates monitoring live Ethereum transactions for fraud patterns.
    """)
    
    # Control panel
    col1, col2, col3 = st.columns(3)
    with col1:
        monitoring_active = st.checkbox("🔴 Start Monitoring", value=False)
    with col2:
        update_interval = st.selectbox("Update Interval", [1, 5, 10, 30], index=1)
    with col3:
        alert_threshold = st.slider("Alert Threshold", 0.0, 1.0, 0.8, 0.05)
    
    # Live metrics display
    if monitoring_active:
        metrics_container = st.container()
        chart_container = st.container()
        alerts_container = st.container()
        
        # Simulate live data
        if 'live_data' not in st.session_state:
            st.session_state.live_data = []
            st.session_state.alert_count = 0
        
        # Generate new transaction
        new_transaction = _generate_simulated_transaction()
        st.session_state.live_data.append(new_transaction)
        
        # Keep only last 50 transactions
        if len(st.session_state.live_data) > 50:
            st.session_state.live_data = st.session_state.live_data[-50:]
        
        # Display metrics
        with metrics_container:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Transactions Processed", len(st.session_state.live_data))
            with col2:
                recent_fraud = sum(1 for tx in st.session_state.live_data[-10:] if tx['fraud_prob'] > alert_threshold)
                st.metric("Recent High Risk", recent_fraud)
            with col3:
                avg_risk = np.mean([tx['fraud_prob'] for tx in st.session_state.live_data])
                st.metric("Average Risk", f"{avg_risk:.2%}")
            with col4:
                st.metric("Total Alerts", st.session_state.alert_count)
        
        # Display chart
        with chart_container:
            if st.session_state.live_data:
                chart_df = pd.DataFrame(st.session_state.live_data)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=chart_df.index,
                    y=chart_df['fraud_prob'],
                    mode='lines+markers',
                    name='Fraud Probability',
                    line=dict(color='red', width=2)
                ))
                fig.add_hline(y=alert_threshold, line_dash="dash", line_color="orange")
                fig.update_layout(
                    title="Real-Time Fraud Probability",
                    xaxis_title="Transaction Number",
                    yaxis_title="Fraud Probability",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Check for alerts
        if new_transaction['fraud_prob'] > alert_threshold:
            st.session_state.alert_count += 1
            with alerts_container:
                st.error(f"🚨 FRAUD ALERT: Address {new_transaction['address']} - Risk: {new_transaction['fraud_prob']:.2%}")
        
        # Auto-refresh
        time.sleep(update_interval)
        st.rerun()

def _render_alert_dashboard_tab():
    """Render alert dashboard interface."""
    st.header("Fraud Alert Dashboard")
    
    st.markdown("""
    ### Alert Management and Monitoring
    
    Central dashboard for managing fraud alerts and investigating suspicious activities.
    """)
    
    # Generate sample alerts
    if 'alerts' not in st.session_state:
        st.session_state.alerts = _generate_sample_alerts()
    
    # Alert summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_alerts = len(st.session_state.alerts)
        st.metric("Total Alerts", total_alerts)
    with col2:
        pending_alerts = len([a for a in st.session_state.alerts if a['status'] == 'Pending'])
        st.metric("Pending Review", pending_alerts)
    with col3:
        high_priority = len([a for a in st.session_state.alerts if a['priority'] == 'High'])
        st.metric("High Priority", high_priority)
    with col4:
        today_alerts = len([a for a in st.session_state.alerts if a['timestamp'].date() == datetime.now().date()])
        st.metric("Today's Alerts", today_alerts)
    
    # Alert filters
    st.subheader("Alert Filters")
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox("Status", ["All", "Pending", "Investigating", "Resolved", "False Positive"])
    with col2:
        priority_filter = st.selectbox("Priority", ["All", "High", "Medium", "Low"])
    with col3:
        time_filter = st.selectbox("Time Range", ["All", "Last 24h", "Last 7d", "Last 30d"])
    
    # Filter alerts
    filtered_alerts = st.session_state.alerts.copy()
    if status_filter != "All":
        filtered_alerts = [a for a in filtered_alerts if a['status'] == status_filter]
    if priority_filter != "All":
        filtered_alerts = [a for a in filtered_alerts if a['priority'] == priority_filter]
    
    # Alert table
    st.subheader("Active Alerts")
    if filtered_alerts:
        alert_df = pd.DataFrame(filtered_alerts)
        alert_df['timestamp'] = pd.to_datetime(alert_df['timestamp'])
        alert_df = alert_df.sort_values('timestamp', ascending=False)
        
        # Color code by priority
        def color_priority(val):
            if val == 'High':
                return 'background-color: #ffcccb'
            elif val == 'Medium':
                return 'background-color: #fff2cc'
            else:
                return 'background-color: #e7f3ff'
        
        styled_alerts = alert_df.style.applymap(color_priority, subset=['priority'])
        st.dataframe(styled_alerts, use_container_width=True)
        
        # Alert actions
        st.subheader("Alert Actions")
        selected_alert = st.selectbox("Select Alert for Action", 
                                    [f"{a['id']} - {a['address']}" for a in filtered_alerts])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔍 Investigate"):
                st.success("Alert marked for investigation")
        with col2:
            if st.button("✅ Mark Resolved"):
                st.success("Alert marked as resolved")
        with col3:
            if st.button("❌ False Positive"):
                st.success("Alert marked as false positive")
    else:
        st.info("No alerts match the current filters.")

def _simulate_fraud_prediction(features):
    """Simulate fraud prediction for given features."""
    # Simple heuristic-based simulation
    sent_tx = features[0] if len(features) > 0 else 0
    received_tx = features[1] if len(features) > 1 else 0
    total_sent = features[2] if len(features) > 2 else 0
    total_received = features[3] if len(features) > 3 else 0
    account_age = features[4] if len(features) > 4 else 1
    
    # Calculate risk factors
    risk_score = 0.0
    risk_factors = []
    
    # High transaction frequency for new accounts
    if account_age < 30 and (sent_tx + received_tx) > 20:
        risk_score += 0.3
        risk_factors.append("High activity for new account")
    
    # Unusual send/receive ratio
    total_tx = sent_tx + received_tx
    if total_tx > 0:
        send_ratio = sent_tx / total_tx
        if send_ratio > 0.9 or send_ratio < 0.1:
            risk_score += 0.25
            risk_factors.append("Unusual transaction direction ratio")
    
    # Large value movements
    if total_sent > 100 or total_received > 100:
        risk_score += 0.2
        risk_factors.append("High value transactions")
    
    # Very low activity
    if total_tx < 2 and account_age > 100:
        risk_score += 0.15
        risk_factors.append("Unusually low activity for account age")
    
    # Add some randomness
    risk_score += np.random.normal(0, 0.1)
    risk_score = max(0, min(1, risk_score))  # Clamp between 0 and 1
    
    return risk_score, risk_factors

def _get_risk_level(fraud_probability):
    """Convert fraud probability to risk level."""
    if fraud_probability >= 0.7:
        return "High Risk"
    elif fraud_probability >= 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"

def _display_fraud_analysis_results(address, fraud_probability, risk_factors, features):
    """Display fraud analysis results."""
    st.subheader("Analysis Results")
    
    # Risk score display
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_level = _get_risk_level(fraud_probability)
        st.metric("Risk Level", risk_level)
    with col2:
        st.metric("Fraud Probability", f"{fraud_probability:.2%}")
    with col3:
        confidence = min(0.95, 0.7 + abs(fraud_probability - 0.5))
        st.metric("Confidence", f"{confidence:.1%}")
    
    # Risk visualization
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = fraud_probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fraud Risk Score"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk factors
    if risk_factors:
        st.subheader("Risk Factors Detected")
        for factor in risk_factors:
            st.warning(f"⚠️ {factor}")
    else:
        st.success("✅ No significant risk factors detected")
    
    # Feature analysis
    with st.expander("📊 Feature Analysis"):
        feature_names = [
            "Sent Transactions", "Received Transactions", "Total Ether Sent", 
            "Total Ether Received", "Account Age (days)", "Unique Recipients",
            "Contract Transactions", "Avg Time Between Tx", "Gas Price Avg"
        ]
        
        feature_df = pd.DataFrame({
            'Feature': feature_names[:len(features)],
            'Value': features[:len(feature_names)]
        })
        st.dataframe(feature_df)

def _generate_simulated_transaction():
    """Generate a simulated transaction for live monitoring."""
    hex_chars = list('0123456789abcdef')
    return {
        'address': f"0x{''.join(np.random.choice(hex_chars, size=8))}...",
        'fraud_prob': np.random.beta(2, 8),  # Skewed towards lower probabilities
        'timestamp': datetime.now(),
        'value': np.random.exponential(5),
        'gas_price': np.random.normal(20, 5)
    }

def _generate_sample_alerts():
    """Generate sample alerts for the dashboard."""
    alerts = []
    hex_chars = list('0123456789abcdef')
    
    for i in range(20):
        alert_time = datetime.now() - timedelta(hours=int(np.random.randint(0, 48)))
        alerts.append({
            'id': f"ALERT_{i+1:03d}",
            'address': f"0x{''.join(np.random.choice(hex_chars, size=8))}...",
            'fraud_probability': np.random.uniform(0.7, 0.99),
            'priority': np.random.choice(['High', 'Medium', 'Low'], p=[0.3, 0.5, 0.2]),
            'status': np.random.choice(['Pending', 'Investigating', 'Resolved', 'False Positive'], 
                                    p=[0.4, 0.3, 0.2, 0.1]),
            'timestamp': alert_time,
            'alert_type': np.random.choice(['High Transaction Volume', 'Unusual Pattern', 
                                         'Suspicious Timing', 'Value Anomaly'])
        })
    return alerts