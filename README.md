# Detecting and Classifying Fraudulent Ethereum Accounts Using Machine Learning

## Academic Context
This project represents the culmination of my undergraduate research in BSc. Data Science and Analytics at Jomo Kenyatta University of Agriculture and Technology (JKUAT). It serves as the final year project demonstrating the application of machine learning techniques to blockchain security challenges.

**Student:** Victor Oketch Sabare  
**Registration Number:** SCT213-C002-0061/2021  
**University:** Jomo Kenyatta University of Agriculture and Technology  
**School:** School of Computing and Information Technology  
**Department:** Department of Computing  
**Academic Supervisor:** Professor Isaac Kega  
**Degree Program:** Bachelor of Science in Data Science and Analytics  
**Project Period:** 2023-2024  

## Project Overview
This research develops a comprehensive machine learning system for detecting fraudulent activities on the Ethereum blockchain by analyzing on-chain transaction patterns. The project employs both supervised and unsupervised techniques to identify anomalous behaviors indicative of fraud, providing a robust framework for cryptocurrency security.

### Research Problem
Fraud within cryptocurrency ecosystems represents a significant impediment to mainstream blockchain adoption, with annual losses exceeding $14 billion globally. This research addresses the challenge of automatically identifying malicious actors on the Ethereum network through their behavioral signatures, transaction patterns, and network interactions.

### Research Objectives
1. To develop feature engineering techniques that effectively capture transaction behaviors associated with fraudulent activities
2. To implement and evaluate unsupervised clustering algorithms for anomaly detection in Ethereum transaction patterns
3. To build supervised classification models that accurately distinguish between legitimate and fraudulent accounts
4. To create a real-time capable detection system with visualization and analysis capabilities
5. To assess the interpretability of fraud detection models through feature importance analysis

## Methodology
The project follows a structured machine learning pipeline approach:

### 1. Data Collection and Preprocessing
- **Data Sources:** Ethereum blockchain transactions, labeled fraud datasets from blockchain security firms
- **Feature Extraction:** 37 distinct features across five categories (temporal, value, network, token, contract)
- **Data Processing:** Handling imbalanced classes, feature scaling, outlier treatment, missing value imputation
- **Dataset Characteristics:** 9,841 labeled Ethereum addresses (2,179 fraudulent, 7,662 legitimate)

### 2. Unsupervised Learning Approach
The project implements eight distinct clustering algorithms to identify natural groupings of transaction behaviors:

- **K-Means:** Baseline clustering using centroid-based partitioning
- **OPTICS:** (Ordering Points To Identify Clustering Structure) Density-based clustering with variable density sensitivity
- **DBSCAN:** (Density-Based Spatial Clustering of Applications with Noise) Identifies high-density regions
- **Hierarchical Clustering:** Builds nested clusters through agglomerative methods
- **Gaussian Mixture Models:** Probabilistic models for soft clustering assignments
- **BIRCH:** (Balanced Iterative Reducing and Clustering using Hierarchies) Efficient clustering for large datasets
- **Isolation Forest:** Anomaly detection through recursive partitioning isolation
- **Affinity Propagation:** Exemplar-based clustering using message passing

Each algorithm is evaluated on its ability to separate fraudulent accounts and computational efficiency.

### 3. Supervised Classification Approach
Seven supervised learning models are implemented to classify accounts as fraudulent or legitimate:

- **XGBoost:** Gradient boosting framework optimized for performance
- **LightGBM:** High-efficiency gradient boosting decision tree framework
- **Stacking Ensemble:** Meta-model combining multiple base classifiers
- **Support Vector Machine (SVM):** Constructs optimal hyperplanes for classification
- **TabNet:** Deep learning approach for tabular data with attention mechanisms
- **Multi-Layer Perceptron (MLP):** Neural network architecture for complex pattern recognition
- **Random Forest:** Ensemble of decision trees with feature importance metrics

Models are evaluated using stratified cross-validation with metrics including accuracy, precision, recall, F1-score, and ROC-AUC.

### 4. Feature Importance Analysis
- **Built-in Tree Importance:** Feature rankings from tree-based algorithms
- **SHAP (SHapley Additive exPlanations):** Game theoretic approach for feature attribution
- **Permutation Importance:** Measuring performance decline when features are shuffled
- **Partial Dependence Plots:** Visualizing feature impact on predictions
- **Cross-category Feature Analysis:** Understanding which feature types most strongly indicate fraud

### 5. Visualization and Interpretation
- Interactive dashboard for model exploration and performance comparison
- Cluster visualization with dimensionality reduction techniques (PCA, t-SNE)
- ROC curves and precision-recall analysis for model evaluation
- Feature importance visualization across different model architectures

## Key Features and Innovations

- **Transaction pattern analysis:** Temporal, value, and network feature extraction from raw blockchain data
- **Account behavior profiling:** Statistical characterization of transaction behaviors
- **Network relationship mapping:** Analysis of interaction patterns between Ethereum addresses
- **Hybrid ML approach:** Integration of supervised and unsupervised techniques for robust detection
- **Multi-model comparison framework:** Standardized evaluation of diverse algorithm families
- **Performance analytics dashboard:** Interactive exploration of model performance and feature importance
- **Real-time detection capabilities:** Optimized inference for low-latency fraud monitoring
- **Interpretability mechanisms:** Explanation components for fraud classification decisions

## Technology Stack
- **Programming Language**: Python 3.8+
- **ML Libraries**: scikit-learn, XGBoost, LightGBM, TensorFlow, PyTorch
- **Blockchain Integration**: Web3.py, Etherscan API
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Streamlit
- **Development Environment**: Jupyter Notebooks, VS Code, Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sabare12/ethereum-fraud-detection.git
cd ethereum-fraud-detection
```

2. Set up the virtual environment:
```bash
python -m venv venv
# For Windows:
venv\Scripts\activate
# For Unix or MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

## Usage

1. Data Collection:
```bash
python src/data/collect_data.py
```

2. Preprocessing:
```bash
python src/data/preprocess.py
```

3. Training Models:
```bash
python src/models/train.py
```

4. Running Detection:
```bash
python src/models/detect.py
```

5. Running the Streamlit Interface:

### a) UI version under `ui/`  
```bash
streamlit run ui/app.py
```

### b) (Legacy) top-level app  
```bash
streamlit run app.py
```

6. Unsupervised Analysis:
Open and run any notebook under `Unsupervised ML Models/` to explore clustering-based anomaly detection.

7. (Optional) Models scripts:
- **Train & save** models (if implemented):  
  ```bash
  python models/save_models.py
  ```  
- **Predict** using saved models (if implemented):  
  ```bash
  python models/predict.py
  ```

## Model Performance
- Accuracy: 85%+
- False Positive Rate: <5%
- Detection Speed: <2s per transaction

## Contributing
1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

## License
MIT License

## Author
Victor Oketch Sabare  
Jomo Kenyatta University of Agriculture and Technology

## Acknowledgments
- Professor Isaac Kega (Project Supervisor)
- JKUAT School of Computing and Information Technology
- Ethereum Developer Community

## Contact
- Email: sabarevictor@gmail.com
- GitHub: https://github.com/sabare12