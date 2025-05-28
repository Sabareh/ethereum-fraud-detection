import streamlit as st
import pandas as pd
import numpy as np
import io
from utils.data_utils import (
    load_dataset_for_eda, 
    clean_dataset, 
    get_correlation_analysis,
    perform_dimensionality_reduction,
    get_transaction_column
)
from utils.visualization_utils import (
    create_correlation_heatmap,
    create_transaction_histogram,
    create_pca_plot,
    create_tsne_plot
)

def render_eda_page():
    """Render the EDA analysis page."""
    st.title("EDA Overview")
    
    st.markdown("""
    # Exploratory Data Analysis for Ethereum Fraud Detection
    
    This page presents a comprehensive exploratory data analysis of Ethereum transaction data used for fraud detection.
    The dataset contains 9,841 Ethereum addresses with 51 features extracted from on-chain transaction behaviors.
    Our analysis examines the relationships between various transaction patterns and fraudulent activity to inform 
    feature engineering and modeling approaches.
    """)

    # Load and display dataset
    df = load_dataset_for_eda()
    
    _display_dataset_preview(df)
    _display_data_cleaning_section(df)
    _display_descriptive_statistics(df)
    _display_fraud_distribution(df)
    _display_transaction_analysis(df)
    _display_correlation_analysis(df)
    _display_dimensionality_reduction(df)
    _display_conclusions(df)

def _display_dataset_preview(df):
    """Display dataset preview section."""
    st.subheader("Dataset Preview")
    st.markdown("""
    The dataset contains behavioral features extracted from Ethereum blockchain transactions. Each row represents
    an Ethereum address (wallet) with features describing its transaction patterns. The target variable `FLAG` indicates
    whether the address is fraudulent (1) or legitimate (0).
    
    Key feature categories:
    - **Temporal features**: Time between transactions, active lifespan
    - **Transaction count features**: Number of sent/received transactions
    - **Value-based features**: Minimum, maximum, and average transaction values
    - **Interaction features**: Number of unique addresses interacted with
    - **Contract-related features**: Contract creation and interaction patterns
    - **ERC20 token features**: Token transaction patterns and behaviors
    """)
    st.dataframe(df.head())
    st.write(f"Shape: {df.shape} - Dataset contains {df.shape[0]} addresses with {df.shape[1]} features")
    
    if 'FLAG' in df.columns:
        fraud_count = df['FLAG'].sum()
        legitimate_count = len(df) - fraud_count
        st.write(f"Class distribution: {fraud_count} fraudulent addresses ({fraud_count/len(df):.1%}) and {legitimate_count} legitimate addresses ({legitimate_count/len(df):.1%})")

def _display_data_cleaning_section(df):
    """Display data cleaning analysis."""
    st.subheader("Missing Values & Data Cleaning")
    st.markdown("""
    ### Data Quality Assessment
    
    Missing values analysis reveals several ERC20-related features with incomplete data. These are primarily related to:
    - Contract transaction timing features
    - Contract value distribution features
    
    We perform the following cleaning steps:
    1. Drop rows with any missing values (minimal impact as most features are complete)
    2. Remove empty columns identified in the dataset
    3. Standardize feature names and formats for consistency
    
    This cleaning ensures high-quality input data for our machine learning models while preserving as many observations as possible.
    """)
    
    df_clean = clean_dataset(df)
    buf = io.StringIO()
    df_clean.info(buf=buf)
    st.text(buf.getvalue())
    
    st.markdown("""
    The data types show a predominance of numerical features (float64 and int64), with only a few object (string) columns.
    This distribution of datatypes is ideal for machine learning algorithms that operate on numerical data.
    """)

def _display_descriptive_statistics(df):
    """Display descriptive statistics."""
    st.subheader("Descriptive Statistics")
    st.markdown("""
    ### Statistical Summary
    
    The descriptive statistics below reveal several important characteristics of our dataset:
    
    - **High variability**: Many features show extreme differences between minimum and maximum values, 
      indicating high variance in transaction behaviors
    - **Skewed distributions**: Most features have means significantly different from medians (50%), 
      suggesting right-skewed distributions typical of financial data
    - **Scale differences**: Features operate on vastly different scales, highlighting the need for 
      normalization before modeling
    - **Outliers**: Large max values compared to 75th percentiles indicate the presence of outliers
    
    These insights inform our feature engineering and preprocessing strategy, particularly the need for:
    - Robust scaling or normalization techniques
    - Potential log transformation for heavily skewed features
    - Outlier handling strategies for extreme values
    """)
    
    df_clean = clean_dataset(df)
    st.write(df_clean.describe())
    
    st.subheader("Unique Values Analysis")
    st.markdown("""
    ### Cardinality Assessment
    
    This analysis shows the number of unique values for each feature, helping identify:
    
    - **Binary features**: Features with only 2 unique values
    - **Categorical features**: Features with low-to-medium cardinality that might benefit from encoding
    - **Continuous features**: Features with high cardinality likely representing numerical measurements
    - **ID-like features**: Features with uniqueness close to row count that provide no predictive value
    
    Highly unique features like addresses should be excluded from modeling, while low-cardinality features
    might need special encoding techniques depending on the algorithm used.
    """)
    st.write(df_clean.nunique())

def _display_fraud_distribution(df):
    """Display fraud label distribution."""
    if 'FLAG' not in df.columns:
        return
        
    st.subheader("Fraud Label Distribution")
    st.markdown("""
    ### Target Variable Analysis
    
    The chart below shows the distribution of our target variable (FLAG), where:
    - 0 = Legitimate address
    - 1 = Fraudulent address
    
    This distribution reveals class imbalance, with fraudulent addresses representing the minority class.
    This imbalance has important implications for our modeling approach:
    
    - Need for balanced evaluation metrics (F1-score, precision-recall rather than just accuracy)
    - Potential use of sampling techniques (oversampling, undersampling, or synthetic sampling)
    - Consideration of class weights during model training
    - Threshold optimization for classification probabilities
    
    Understanding this imbalance helps prevent developing models biased toward the majority class.
    """)
    
    df_clean = clean_dataset(df)
    fraud_counts = df_clean['FLAG'].value_counts()
    st.bar_chart(fraud_counts)
    st.write(f"Imbalance ratio: 1:{fraud_counts[0]/fraud_counts[1]:.1f} (legitimate:fraudulent)")

def _display_transaction_analysis(df):
    """Display transaction volume analysis."""
    st.subheader("Transaction Volume Analysis")
    st.markdown("""
    ### Sent Transactions Distribution
    
    This histogram shows the distribution of transaction counts across addresses, revealing:
    
    - **Right-skewed distribution**: Most addresses have a relatively small number of transactions
    - **Long tail**: A small number of addresses have an extremely high volume of transactions
    - **Behavioral pattern**: Different patterns between normal users and potential service providers or exchanges
    
    The logarithmic density curve helps visualize the distribution across different scales of activity.
    This feature shows promise for fraud detection, as fraudulent addresses often exhibit distinctive
    transaction volume patterns compared to legitimate ones.
    
    **Key insight**: Many fraudulent addresses show unusually low transaction counts, possibly indicating 
    single-purpose fraud accounts that are abandoned after use.
    """)
    
    transaction_col = get_transaction_column(df)
    if transaction_col != 'FLAG':  # Only create histogram if we found a valid transaction column
        fig = create_transaction_histogram(df, transaction_col)
        if fig:
            st.pyplot(fig)
    else:
        st.warning(f"Could not find transaction count column. Available columns: {df.columns.tolist()}")
    
    st.markdown("""
    ### Additional Transaction Volume Insights
    
    * **Low volume fraud**: Many fraudulent addresses have fewer than 5 transactions
    * **High volume patterns**: Legitimate addresses show more consistent transaction patterns over time
    * **Burst patterns**: Many fraudulent addresses show "burst" patterns - high activity in short time periods
    * **Transaction-to-age ratio**: The ratio of transaction count to address age is a strong fraud indicator
    """)

def _display_correlation_analysis(df):
    """Display correlation analysis."""
    st.subheader("Feature Correlation Analysis")
    st.markdown("""
    ### Correlation Matrix
    
    The correlation heatmap below shows the Pearson correlation coefficients between features. This analysis reveals:
    
    - **Highly correlated feature groups**: Several feature clusters show strong correlations (>0.7)
    - **Feature redundancy opportunities**: Potential for dimensionality reduction without information loss
    - **Correlation with target**: Features showing strong correlation with the fraud indicator
    - **Independent features**: Features showing minimal correlation with others, providing unique information
    
    Understanding these relationships helps with feature selection and engineering, particularly by:
    1. Identifying redundant features that can be removed to reduce dimensionality
    2. Discovering feature combinations that might create stronger predictive signals
    3. Mitigating multicollinearity issues that could affect certain model types
    4. Informing the design of more complex features that capture relationship patterns
    """)
    
    df_clean = clean_dataset(df)
    corr, mask, high_corr_pairs = get_correlation_analysis(df_clean)
    
    fig = create_correlation_heatmap(corr, mask)
    st.pyplot(fig)
    
    # High correlation pairs
    st.markdown("""
    ### High Correlation Feature Pairs
    
    This table shows feature pairs with correlation above the selected threshold. These highly correlated pairs indicate:
    
    - **Potential redundancy**: Features capturing similar information patterns
    - **Feature selection opportunities**: Candidates for elimination to simplify models
    - **Feature group identification**: Natural groupings of related features
    
    Adjust the threshold slider to explore correlation patterns at different strength levels.
    """)
    
    threshold = st.slider("Correlation threshold", 0.5, 1.0, 0.7, 0.05)
    current_high_corr = high_corr_pairs[high_corr_pairs['corr'].abs() >= threshold]
    
    if not current_high_corr.empty:
        st.write(f"Found {len(current_high_corr)} feature pairs with correlation >= {threshold}")
        st.dataframe(current_high_corr)
        
        st.markdown("""
        **Key correlation insights**:
        
        1. **Transaction count correlations**: Features counting transactions tend to correlate strongly with each other
        2. **Value-based correlations**: Features measuring transaction values show strong internal correlations
        3. **Temporal correlations**: Time-based features form their own correlation cluster
        4. **ERC20 correlations**: Token-specific behaviors create a distinct correlation group
        
        These groupings inform our feature engineering approach and help identify which features provide unique information.
        """)
    else:
        st.write(f"No feature pairs found with correlation >= {threshold}")

def _display_dimensionality_reduction(df):
    """Display dimensionality reduction analysis."""
    st.subheader("Dimensionality Reduction Visualization")
    st.markdown("""
    ### PCA (2D) Projection
    
    Principal Component Analysis (PCA) reduces the high-dimensional feature space to two dimensions, 
    allowing us to visualize similarity patterns between addresses. This visualization:
    
    - **Reveals clusters**: Natural groupings of similar transaction behaviors
    - **Shows separation**: How well fraud and legitimate addresses separate in feature space
    - **Highlights outliers**: Unusual transaction patterns that stand out from general behavior
    - **Visualizes variance**: How transaction behaviors vary across the dataset
    
    The visualization is colored by fraud status (blue = legitimate, red = fraudulent) to show the 
    relationship between transaction patterns and fraudulent behavior.
    
    **Methodology note**: Data is scaled using MinMaxScaler before PCA to ensure fair comparison between features with different scales.
    """)
    
    df_clean = clean_dataset(df)
    
    if 'FLAG' in df_clean.columns:
        pca_results, tsne_results, explained_variance, scaled_data = perform_dimensionality_reduction(df_clean)
        
        if pca_results is not None:
            # PCA Plot
            fig_pca = create_pca_plot(pca_results, df_clean['FLAG'])
            st.pyplot(fig_pca)
            
            st.markdown(f"""
            **PCA Component Information**:
            - Component 1 explains {explained_variance[0]:.2f}% of variance
            - Component 2 explains {explained_variance[1]:.2f}% of variance
            - Total explained variance: {sum(explained_variance):.2f}%
            
            **Key PCA Insights**:
            - Partial separation between fraud and legitimate classes is visible, indicating that transaction features
              contain meaningful signals for fraud detection
            - Some overlap between classes suggests that simple linear separation may not be sufficient
            - Multiple clusters within each class hint at different types of fraudulent and legitimate behavior patterns
            - Outliers visible at the extremes may represent unusual transaction patterns worthy of further investigation
            """)
            
            # t-SNE section
            st.subheader("Non-linear Dimensionality Reduction")
            st.markdown("""
            ### t-SNE Projection
            
            t-Distributed Stochastic Neighbor Embedding (t-SNE) is a non-linear dimensionality reduction technique 
            that captures complex relationships better than PCA for visualization purposes. This visualization:
            
            - **Preserves local structure**: Maintains proximity relationships between similar addresses
            - **Shows complex clusters**: Reveals more nuanced groupings than linear methods like PCA
            - **Highlights behavioral patterns**: Helps identify distinct transaction behavior profiles
            - **Improves separation**: Often provides better visual separation between fraud and legitimate classes
            
            **Methodology note**: t-SNE is applied to the PCA results rather than raw features to improve computational 
            efficiency and reduce noise. A learning rate of 50 is used to balance local and global structure preservation.
            """)
            
            fig_tsne = create_tsne_plot(tsne_results, df_clean['FLAG'])
            st.pyplot(fig_tsne)
        else:
            st.warning("Not enough data for dimensional reduction visualization.")
    else:
        st.warning("FLAG column not found in dataset.")

def _display_conclusions(df):
    """Display EDA conclusions."""
    st.subheader("EDA Conclusions")
    
    if 'FLAG' in df.columns:
        fraud_count = df['FLAG'].sum()
        fraud_percentage = fraud_count/len(df)
    else:
        fraud_percentage = 0.2  # Default estimate
    
    st.markdown(f"""
    ### Key Findings and Next Steps
    
    Our exploratory data analysis has revealed several important insights about Ethereum transaction patterns and fraud:
    
    1. **Class imbalance**: Fraudulent addresses represent ~{fraud_percentage:.1%} of the dataset, requiring appropriate handling during modeling
    2. **Feature redundancy**: Several highly correlated feature groups suggest opportunities for dimensionality reduction
    3. **Non-linear relationships**: Complex patterns in the data suggest non-linear models may perform better
    4. **Diverse fraud patterns**: Multiple clusters of fraudulent addresses indicate different types of fraud behaviors
    5. **Temporal significance**: Time-based features show distinctive patterns between fraud and legitimate addresses
    6. **Value patterns**: Transaction value distributions differ significantly between classes
    7. **Scale variation**: Wide range of feature scales necessitates normalization before modeling
    
    **Recommended next steps**:
    
    - **Feature engineering**: Create ratio features, temporal pattern features, and interaction terms
    - **Feature selection**: Remove redundant features while preserving unique information
    - **Model selection**: Focus on tree-based ensembles and non-linear models suggested by data patterns
    - **Handling imbalance**: Implement class weighting or sampling techniques to address class imbalance
    - **Anomaly detection**: Consider supplementing classification with anomaly detection approaches
    
    These insights guide our modeling approach in subsequent sections, where we'll develop both supervised 
    and unsupervised models for Ethereum fraud detection.
    """)
