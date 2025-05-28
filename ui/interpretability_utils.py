import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Try to import optional interpretability libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not available. Install with: pip install lime")

try:
    import eli5
    from eli5 import show_weights
    ELI5_AVAILABLE = True
except ImportError:
    ELI5_AVAILABLE = False
    print("ELI5 not available. Install with: pip install eli5")


class ModelExplainer:
    """
    Comprehensive model explainability class providing multiple interpretation methods.
    """
    
    def __init__(self, model, X_train, X_test, y_train, y_test, feature_names=None):
        """
        Initialize the explainer with a trained model and data.
        
        Parameters:
        -----------
        model : sklearn model
            Trained machine learning model
        X_train, X_test : array-like
            Training and test features
        y_train, y_test : array-like
            Training and test labels
        feature_names : list, optional
            Names of features for better visualization
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(X_train.shape[1])]
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        
        if SHAP_AVAILABLE:
            self._initialize_shap()
        if LIME_AVAILABLE:
            self._initialize_lime()
    
    def _initialize_shap(self):
        """Initialize SHAP explainer based on model type."""
        try:
            # Try TreeExplainer for tree-based models
            if hasattr(self.model, 'tree_') or hasattr(self.model, 'estimators_'):
                self.shap_explainer = shap.TreeExplainer(self.model)
            else:
                # Use KernelExplainer for other models (slower but works with any model)
                background = shap.sample(self.X_train, 100)  # Use sample for efficiency
                self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, background)
        except Exception as e:
            print(f"Could not initialize SHAP TreeExplainer: {e}")
            try:
                # Fallback to KernelExplainer
                background = shap.sample(self.X_train, 50)
                self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, background)
            except Exception as e2:
                print(f"Could not initialize SHAP KernelExplainer: {e2}")
                self.shap_explainer = None
    
    def _initialize_lime(self):
        """Initialize LIME explainer."""
        try:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_train,
                feature_names=self.feature_names,
                class_names=['Legitimate', 'Fraudulent'],
                mode='classification'
            )
        except Exception as e:
            print(f"Could not initialize LIME explainer: {e}")
            self.lime_explainer = None
    
    def get_feature_importance(self, method='builtin'):
        """
        Get feature importance using different methods.
        
        Parameters:
        -----------
        method : str
            Method to use: 'builtin', 'permutation', 'shap'
        
        Returns:
        --------
        pd.DataFrame
            Feature importance scores
        """
        if method == 'builtin' and hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        
        elif method == 'permutation':
            perm_importance = permutation_importance(
                self.model, self.X_test, self.y_test, 
                n_repeats=10, random_state=42
            )
            importance = perm_importance.importances_mean
        
        elif method == 'shap' and self.shap_explainer:
            try:
                shap_values = self.shap_explainer.shap_values(self.X_test[:100])  # Sample for efficiency
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Use positive class for binary classification
                importance = np.abs(shap_values).mean(0)
            except Exception as e:
                print(f"Error computing SHAP importance: {e}")
                return None
        
        else:
            print(f"Method '{method}' not available for this model")
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, method='builtin', top_n=20):
        """
        Plot feature importance.
        
        Parameters:
        -----------
        method : str
            Method to use for importance calculation
        top_n : int
            Number of top features to display
        
        Returns:
        --------
        matplotlib.figure.Figure
            Feature importance plot
        """
        importance_df = self.get_feature_importance(method)
        if importance_df is None:
            return None
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        
        top_features = importance_df.head(top_n)
        
        bars = ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel(f'Feature Importance ({method})')
        ax.set_title(f'Top {top_n} Most Important Features')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        return fig
    
    def get_shap_values(self, sample_size=100):
        """
        Get SHAP values for interpretation.
        
        Parameters:
        -----------
        sample_size : int
            Number of samples to compute SHAP values for
        
        Returns:
        --------
        numpy.ndarray
            SHAP values
        """
        if not self.shap_explainer:
            print("SHAP explainer not available")
            return None
        
        try:
            sample_data = self.X_test[:sample_size]
            shap_values = self.shap_explainer.shap_values(sample_data)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Binary classification: return positive class values
                return shap_values[1]
            else:
                # Single output
                return shap_values
        except Exception as e:
            print(f"Error computing SHAP values: {e}")
            return None
    
    def plot_shap_summary(self, sample_size=100):
        """
        Create SHAP summary plot.
        
        Parameters:
        -----------
        sample_size : int
            Number of samples to use
        
        Returns:
        --------
        matplotlib.figure.Figure
            SHAP summary plot
        """
        if not SHAP_AVAILABLE or not self.shap_explainer:
            print("SHAP not available")
            return None
        
        shap_values = self.get_shap_values(sample_size)
        if shap_values is None:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        try:
            shap.summary_plot(
                shap_values, 
                self.X_test[:sample_size], 
                feature_names=self.feature_names,
                show=False
            )
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error creating SHAP summary plot: {e}")
            return None
    
    def plot_shap_waterfall(self, instance_idx=0):
        """
        Create SHAP waterfall plot for a single instance.
        
        Parameters:
        -----------
        instance_idx : int
            Index of instance to explain
        
        Returns:
        --------
        matplotlib.figure.Figure
            SHAP waterfall plot
        """
        if not SHAP_AVAILABLE or not self.shap_explainer:
            print("SHAP not available")
            return None
        
        try:
            shap_values = self.get_shap_values(instance_idx + 1)
            if shap_values is None:
                return None
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create waterfall plot data
            values = shap_values[instance_idx]
            base_value = self.shap_explainer.expected_value
            if isinstance(base_value, list):
                base_value = base_value[1]  # Use positive class
            
            # Sort by absolute impact
            sorted_idx = np.argsort(np.abs(values))[::-1][:15]  # Top 15 features
            
            cumulative = base_value
            y_pos = 0
            
            # Plot base value
            ax.barh(y_pos, base_value, color='gray', alpha=0.7)
            ax.text(base_value/2, y_pos, f'Base: {base_value:.3f}', 
                   ha='center', va='center')
            
            # Plot each feature contribution
            for i, idx in enumerate(sorted_idx):
                feature_name = self.feature_names[idx]
                contribution = values[idx]
                
                y_pos += 1
                color = 'red' if contribution > 0 else 'blue'
                
                ax.barh(y_pos, contribution, left=cumulative, 
                       color=color, alpha=0.7)
                
                # Add text annotation
                text_x = cumulative + contribution/2
                ax.text(text_x, y_pos, f'{feature_name}: {contribution:.3f}', 
                       ha='center', va='center', fontsize=8)
                
                cumulative += contribution
            
            # Final prediction
            y_pos += 1
            ax.barh(y_pos, cumulative, color='green', alpha=0.7)
            ax.text(cumulative/2, y_pos, f'Prediction: {cumulative:.3f}', 
                   ha='center', va='center')
            
            ax.set_xlabel('SHAP Value')
            ax.set_ylabel('Features')
            ax.set_title(f'SHAP Waterfall Plot - Instance {instance_idx}')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error creating SHAP waterfall plot: {e}")
            return None
    
    def explain_instance_lime(self, instance_idx=0, num_features=10):
        """
        Explain a single instance using LIME.
        
        Parameters:
        -----------
        instance_idx : int
            Index of instance to explain
        num_features : int
            Number of features to include in explanation
        
        Returns:
        --------
        dict
            LIME explanation data
        """
        if not LIME_AVAILABLE or not self.lime_explainer:
            print("LIME not available")
            return None
        
        try:
            instance = self.X_test[instance_idx]
            explanation = self.lime_explainer.explain_instance(
                instance, 
                self.model.predict_proba,
                num_features=num_features
            )
            
            # Extract explanation data
            exp_data = explanation.as_list()
            
            return {
                'features': [item[0] for item in exp_data],
                'contributions': [item[1] for item in exp_data],
                'prediction_proba': self.model.predict_proba([instance])[0],
                'prediction': self.model.predict([instance])[0]
            }
            
        except Exception as e:
            print(f"Error creating LIME explanation: {e}")
            return None
    
    def plot_partial_dependence(self, features, n_cols=2):
        """
        Create partial dependence plots.
        
        Parameters:
        -----------
        features : list
            List of feature indices or names to plot
        n_cols : int
            Number of columns in subplot grid
        
        Returns:
        --------
        matplotlib.figure.Figure
            Partial dependence plots
        """
        try:
            # Convert feature names to indices if necessary
            if isinstance(features[0], str):
                feature_indices = [self.feature_names.index(f) for f in features if f in self.feature_names]
            else:
                feature_indices = features
            
            n_features = len(feature_indices)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_features == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i, feature_idx in enumerate(feature_indices):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                
                # Compute partial dependence manually
                try:
                    pd_result = partial_dependence(
                        self.model, self.X_test, [feature_idx], 
                        grid_resolution=50
                    )
                    
                    # Extract values based on sklearn version
                    if hasattr(pd_result, 'average'):
                        # Newer sklearn versions
                        values = pd_result.average[0]
                        grid = pd_result.grid_values[0]
                    else:
                        # Older sklearn versions
                        values = pd_result[0][0]
                        grid = pd_result[1][0]
                    
                    # Plot
                    ax.plot(grid, values)
                    ax.set_xlabel(self.feature_names[feature_idx])
                    ax.set_ylabel('Partial Dependence')
                    ax.set_title(f'Partial Dependence: {self.feature_names[feature_idx]}')
                    ax.grid(True, alpha=0.3)
                    
                except Exception as e:
                    # Fallback: create a simple approximation
                    feature_values = self.X_test[:, feature_idx]
                    min_val, max_val = np.min(feature_values), np.max(feature_values)
                    grid = np.linspace(min_val, max_val, 50)
                    
                    # Create modified data for each grid point
                    pd_values = []
                    sample_data = self.X_test[:100]  # Use subset for efficiency
                    
                    for val in grid:
                        modified_data = sample_data.copy()
                        modified_data[:, feature_idx] = val
                        predictions = self.model.predict_proba(modified_data)[:, 1]
                        pd_values.append(np.mean(predictions))
                    
                    ax.plot(grid, pd_values)
                    ax.set_xlabel(self.feature_names[feature_idx])
                    ax.set_ylabel('Average Prediction')
                    ax.set_title(f'Partial Dependence: {self.feature_names[feature_idx]}')
                    ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(n_features, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                if n_rows > 1:
                    axes[row, col].set_visible(False)
                else:
                    axes[col].set_visible(False)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error creating partial dependence plots: {e}")
            return None
    
    def generate_explanation_report(self, instance_idx=0, save_path=None):
        """
        Generate a comprehensive explanation report for an instance.
        
        Parameters:
        -----------
        instance_idx : int
            Index of instance to explain
        save_path : str, optional
            Path to save the report
        
        Returns:
        --------
        dict
            Comprehensive explanation data
        """
        report = {
            'instance_index': instance_idx,
            'prediction': None,
            'prediction_proba': None,
            'feature_values': None,
            'explanations': {}
        }
        
        # Get basic prediction info
        instance = self.X_test[instance_idx:instance_idx+1]
        report['prediction'] = self.model.predict(instance)[0]
        report['prediction_proba'] = self.model.predict_proba(instance)[0]
        report['feature_values'] = dict(zip(self.feature_names, self.X_test[instance_idx]))
        
        # Get different explanations
        if SHAP_AVAILABLE and self.shap_explainer:
            try:
                shap_values = self.get_shap_values(instance_idx + 1)
                if shap_values is not None:
                    report['explanations']['shap'] = {
                        'values': dict(zip(self.feature_names, shap_values[0])),
                        'base_value': self.shap_explainer.expected_value
                    }
            except Exception as e:
                print(f"Error adding SHAP to report: {e}")
        
        if LIME_AVAILABLE and self.lime_explainer:
            lime_exp = self.explain_instance_lime(instance_idx)
            if lime_exp:
                report['explanations']['lime'] = lime_exp
        
        # Add feature importance
        for method in ['builtin', 'permutation']:
            importance = self.get_feature_importance(method)
            if importance is not None:
                report['explanations'][f'{method}_importance'] = importance.to_dict('records')
        
        return report


def compare_feature_importance_methods(explainer, methods=['builtin', 'permutation', 'shap']):
    """
    Compare feature importance across different methods.
    
    Parameters:
    -----------
    explainer : ModelExplainer
        Initialized explainer object
    methods : list
        List of methods to compare
    
    Returns:
    --------
    pd.DataFrame
        Comparison dataframe
    matplotlib.figure.Figure
        Comparison plot
    """
    importance_data = {}
    
    for method in methods:
        importance_df = explainer.get_feature_importance(method)
        if importance_df is not None:
            importance_data[method] = importance_df.set_index('feature')['importance']
    
    if not importance_data:
        print("No importance data available")
        return None, None
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(importance_data)
    comparison_df = comparison_df.fillna(0)
    
    # Create correlation plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Heatmap of correlations
    corr_matrix = comparison_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax1)
    ax1.set_title('Feature Importance Method Correlations')
    
    # Top features comparison
    top_features = comparison_df.abs().mean(axis=1).nlargest(15).index
    top_comparison = comparison_df.loc[top_features]
    
    top_comparison.plot(kind='bar', ax=ax2)
    ax2.set_title('Top 15 Features - Method Comparison')
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Importance Score')
    ax2.legend(title='Methods')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    return comparison_df, fig


def create_decision_boundary_plot(model, X, y, feature_idx1=0, feature_idx2=1, resolution=100):
    """
    Create a 2D decision boundary plot for two features.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X : array-like
        Feature data
    y : array-like
        Target labels
    feature_idx1, feature_idx2 : int
        Indices of features to plot
    resolution : int
        Grid resolution for boundary
    
    Returns:
    --------
    matplotlib.figure.Figure
        Decision boundary plot
    """
    # Extract the two features
    X_subset = X[:, [feature_idx1, feature_idx2]]
    
    # Create a mesh
    h = 0.02  # step size
    x_min, x_max = X_subset[:, 0].min() - 1, X_subset[:, 0].max() + 1
    y_min, y_max = X_subset[:, 1].min() - 1, X_subset[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Create a temporary model for 2D data
    temp_model = type(model)(**model.get_params() if hasattr(model, 'get_params') else {})
    temp_model.fit(X_subset, y)
    
    # Predict on the mesh
    Z = temp_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    
    # Plot data points
    scatter = ax.scatter(X_subset[:, 0], X_subset[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    
    ax.set_xlabel(f'Feature {feature_idx1}')
    ax.set_ylabel(f'Feature {feature_idx2}')
    ax.set_title('Decision Boundary Visualization')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Class')
    
    return fig
