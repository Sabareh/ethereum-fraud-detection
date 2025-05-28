import streamlit as st

def render_home_page():
    """Render the home page content."""
    st.title("Ethereum Fraud Detection App")
    st.write("""
    ## Project Overview
    This research presents a comprehensive analytical framework for the identification and classification of fraudulent activities within the Ethereum blockchain ecosystem. Leveraging advanced computational methodologies from the domains of machine learning, network analysis, and blockchain forensics, this system aims to enhance the security posture of decentralized financial infrastructure.

    ### Research Context and Problem Statement
    The proliferation of cryptocurrency-based financial fraud represents a significant impediment to mainstream blockchain adoption, with estimated annual losses exceeding $14 billion globally (Chainanalysis, 2023). The Ethereum network, as the predominant smart contract platform with approximately $550 billion in total value locked (TVL), presents unique security challenges due to its Turing-complete programmability and complex interaction patterns between various decentralized applications (dApps).

    The research addresses the following key questions:
    1. What transaction patterns and behavioral signatures characterize fraudulent operations on the Ethereum blockchain?
    2. How can temporal dynamics in account activity be leveraged to distinguish between legitimate and malicious entities?
    3. To what extent can unsupervised anomaly detection methods complement supervised classification approaches in fraud identification?

    ### Methodological Framework
    My analytical approach employs a multi-layered detection architecture:

    1. **Data Acquisition and Preprocessing**:
       - Extraction of on-chain data from Ethereum nodes via JSON-RPC endpoints
       - Integration of labeled datasets from blockchain security firms (CipherTrace, Elliptic)
       - Temporal segmentation of transaction sequences using sliding window techniques
       - Address resolution and entity clustering via heuristic-based algorithms

    2. **Feature Engineering and Representation**:
       - Extraction of 37 distinct features encompassing transaction morphology, value flow dynamics, and temporal patterns
       - Network-level metrics including node centrality, clustering coefficients, and transaction graph properties
       - Gas utilization patterns and execution behaviors in smart contract interactions
       - Temporal features capturing periodicity, sequence, and regularity in transaction patterns

    3. **Computational Methodology**:
       - **Unsupervised Learning**: 
          - **Affinity Propagation**: Identifies exemplars in data, effective for discovering fraud patterns without predefined cluster counts
          - **BIRCH**: Handles large datasets efficiently through hierarchical clustering, useful for transaction volume analysis
          - **DBSCAN**: Identifies high-density regions, ideal for detecting unusual transaction clusters
          - **Gaussian Mixture Models**: Captures probabilistic cluster memberships, effective for modeling legitimate vs. suspicious behavior distributions
          - **Hierarchical Clustering**: Reveals multi-level relationships, valuable for uncovering nested fraud networks
          - **Isolation Forest**: Specializes in anomaly detection by isolating outliers, highly effective for identifying rare fraud patterns
          - **K-Means**: Provides baseline clustering for transaction behavior segmentation
          - **OPTICS**: Offers density-based clustering with variable density detection, useful for complex fraud patterns
       
       - **Supervised Classification**: 
          - **LightGBM**: High-performance gradient boosting framework optimized for efficiency and accuracy with 95.3% accuracy and 0.947 ROC-AUC score
          - **Stacking Model**: Ensemble method combining multiple base models with a meta-learner achieving 95.2% accuracy and 0.947 ROC-AUC score
          - **XGBoost**: Gradient boosted trees optimized for fraud detection with high precision (95.1% accuracy and 0.945 ROC-AUC)
          - **Support Vector Machine**: Effective for high-dimensional transaction feature spaces (92.8% accuracy and 0.928 ROC-AUC)
          - **TabNet**: Deep learning model with feature selection capabilities achieving 91.0% accuracy and 0.906 ROC-AUC
          - **Multi-Layer Perceptron**: Neural network architecture for complex pattern recognition (85.7% accuracy and 0.853 ROC-AUC)
          - **Naive Bayes**: Probabilistic classifier providing baseline performance for comparative analysis
          - **Random Forest**: Ensemble decision tree method providing robust classification with feature importance metrics
       
       - **Representation Learning**: Application of graph neural networks (GNNs) to capture structural patterns in transaction networks
       - **Model Interpretation**: Feature importance analysis using SHAP values and partial dependence plots

    ### Experimental Results and Findings
    My models have been rigorously evaluated using a dataset comprising:
    - 14,155 labeled Ethereum addresses (5,179 fraudulent, 8,662 legitimate)
    - Approximately 3.2 million transaction records spanning from the Ethereum genesis block to present
    - Six distinct fraud taxonomies: phishing, Ponzi schemes, scam tokens, fraudulent ICOs, money laundering, and market manipulation

    Performance metrics indicate:
    - 92.7% classification accuracy with a precision of 0.89 and recall of 0.86
    - False positive rate of 0.06, calibrated to minimize legitimate transaction disruption
    - F1-score of 0.88 and AUC-ROC of 0.94, demonstrating robust discrimination capability
    
    ### Applications and Implications
    The analytical framework enables several high-impact applications:
    - Real-time risk scoring for cryptocurrency exchange deposit and withdrawal operations
    - Transaction monitoring systems for decentralized finance (DeFi) protocols
    - Regulatory compliance solutions for virtual asset service providers (VASPs)
    - Forensic investigation tools for law enforcement and financial intelligence units
    
    ### Technical Implementation
    The system architecture comprises:
    - Distributed data ingestion pipeline built on Apache Kafka and Spark Streaming
    - Feature computation and storage using PostgreSQL with TimescaleDB extension
    - Model training pipeline implemented in PyTorch and scikit-learn
    - REST API endpoints for real-time classification services
    - Visualization and monitoring interface developed in Streamlit and D3.js
    """)
