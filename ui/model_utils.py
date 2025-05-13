import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Unsupervised models
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch, OPTICS, AffinityPropagation
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest

# Supervised models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


def load_data(use_synthetic=False):
    """
    Load Ethereum transaction data for fraud detection analysis.
    
    Parameters:
    -----------
    use_synthetic : bool, default=False
        If True, generate synthetic data instead of loading from files.
        
    Returns:
    --------
    X : array-like
        Features for fraud detection
    y : array-like
        Binary target labels (1 = fraud, 0 = legitimate)
    """
    # Try to load real data first (unless explicitly asked for synthetic)
    if not use_synthetic:
        try:
            # Attempt to load the real dataset
            # Adjust paths as needed for your environment
            data_path = os.path.join("..", "Data", "address_data_combined.csv")
            df = pd.read_csv(data_path)
            
            # Extract features and target
            X = df.drop(['FLAG', 'Address'], errors='ignore').values
            y = df['FLAG'].values
            
            # Basic preprocessing
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            return X, y
            
        except Exception as e:
            if not use_synthetic:
                print(f"Failed to load real data: {str(e)}")
                print("Falling back to synthetic data...")
    
    # Generate synthetic data if requested or if real data failed to load
    print("Generating synthetic data for demonstration purposes...")
    return generate_synthetic_data()


def generate_synthetic_data(n_samples=1000, n_features=37, fraud_ratio=0.22):
    """
    Generate synthetic Ethereum transaction data for demonstration purposes.
    
    Parameters:
    -----------
    n_samples : int, default=1000
        Number of samples (addresses) to generate
    n_features : int, default=37
        Number of features to generate
    fraud_ratio : float, default=0.22
        Ratio of fraudulent addresses in the dataset
    
    Returns:
    --------
    X : array-like of shape (n_samples, n_features)
        Synthetic features
    y : array-like of shape (n_samples,)
        Binary target labels (1 = fraud, 0 = legitimate)
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate binary labels based on fraud ratio
    n_fraud = int(n_samples * fraud_ratio)
    y = np.zeros(n_samples)
    y[:n_fraud] = 1
    np.random.shuffle(y)
    
    # Generate features with different distributions for fraud vs legitimate
    X = np.zeros((n_samples, n_features))
    
    # For each sample
    for i in range(n_samples):
        if y[i] == 1:  # Fraudulent pattern
            # Transaction counts (tend to be lower for fraud)
            X[i, 0:5] = np.random.exponential(scale=3, size=5)
            
            # Transaction values (tend to be more extreme)
            X[i, 5:10] = np.random.exponential(scale=10, size=5)
            
            # Temporal features (short lifespan)
            X[i, 10:15] = np.random.exponential(scale=2, size=5)
            
            # Network features (few connections)
            X[i, 15:20] = np.random.poisson(lam=2, size=5)
            
            # Contract interactions (unusual patterns)
            X[i, 20:25] = np.random.gamma(shape=1, scale=2, size=5)
            
            # Token transactions (suspicious patterns)
            X[i, 25:30] = np.random.gamma(shape=0.5, scale=4, size=5)
            
            # Other features
            X[i, 30:] = np.random.normal(loc=0.2, scale=0.5, size=n_features-30)
            
        else:  # Legitimate pattern
            # Transaction counts (tend to be higher for legitimate)
            X[i, 0:5] = np.random.normal(loc=10, scale=5, size=5)
            
            # Transaction values (more moderate)
            X[i, 5:10] = np.random.normal(loc=5, scale=2, size=5)
            
            # Temporal features (longer lifespan)
            X[i, 10:15] = np.random.normal(loc=10, scale=4, size=5)
            
            # Network features (more connections)
            X[i, 15:20] = np.random.poisson(lam=10, size=5)
            
            # Contract interactions (typical patterns)
            X[i, 20:25] = np.random.normal(loc=5, scale=2, size=5)
            
            # Token transactions (typical patterns)
            X[i, 25:30] = np.random.normal(loc=8, scale=3, size=5)
            
            # Other features
            X[i, 30:] = np.random.normal(loc=0.6, scale=0.3, size=n_features-30)
    
    # Add some noise
    X += np.random.normal(scale=0.1, size=X.shape)
    
    # Ensure non-negative values for count-like features
    X = np.maximum(X, 0)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y


def get_unsupervised_models():
    """
    Get a dictionary of unsupervised learning models for clustering and anomaly detection.
    
    Returns:
    --------
    dict
        Dictionary mapping model names to initialized model objects
    """
    models = {
        "K-Means": KMeans(n_clusters=2, random_state=42),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
        "Hierarchical Clustering": AgglomerativeClustering(n_clusters=2),
        "Gaussian Mixture": GaussianMixture(n_components=2, random_state=42),
        "Isolation Forest": IsolationForest(random_state=42, contamination=0.2),
        "BIRCH": Birch(n_clusters=2),
        "OPTICS": OPTICS(min_samples=5),
        "Affinity Propagation": AffinityPropagation(random_state=42, damping=0.9)
    }
    
    return models


def get_supervised_models():
    """
    Get a dictionary of supervised learning models for classification.
    
    Returns:
    --------
    dict
        Dictionary mapping model names to initialized model objects
    """
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "SVM": SVC(probability=True, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=300, random_state=42)
    }
    
    # Add models that might not be available in all environments
    if XGB_AVAILABLE:
        models["XGBoost"] = xgb.XGBClassifier(random_state=42)
    
    if LGBM_AVAILABLE:
        models["LightGBM"] = lgb.LGBMClassifier(random_state=42)
    
    return models


if __name__ == "__main__":
    # Test the functions
    X, y = load_data(use_synthetic=True)
    print(f"Generated synthetic data with shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    
    # Verify models
    unsupervised_models = get_unsupervised_models()
    print(f"Available unsupervised models: {list(unsupervised_models.keys())}")
    
    supervised_models = get_supervised_models()
    print(f"Available supervised models: {list(supervised_models.keys())}")
