# Comparative Analysis of Supervised Classifiers for Ethereum Fraud Detection

## Abstract
This document presents a systematic comparison of seven supervised learning algorithms applied to Ethereum transaction data for fraud detection. We evaluate model performance using a consistent dataset and standardized metrics, analyze computational efficiency, and discuss trade-offs. Our goal is to inform practitioners on optimal model selection under various operational constraints.

## 1 Introduction
Cryptocurrency networks are vulnerable to fraudulent activities that can cause significant financial loss. Machine learning offers automated detection mechanisms by modeling historical transaction patterns. This study focuses on supervised classifiers trained to predict a binary fraud indicator (`FLAG`), comparing their effectiveness and efficiency.

### 1.1 Objectives
- Quantify predictive performance (accuracy, precision, recall, F1-score, ROC-AUC).  
- Measure computational cost (training + inference time).  
- Provide guidance under differing resource and risk priorities.

## 2 Methodology

### 2.1 Dataset
- **Source**: Aggregated account‐level features from on‐chain Ethereum transactions.  
- **Preprocessing**: Log-transform skewed features, MinMax scale to [0,1].  
- **Split**: Stratified 70% train, 30% test (random seed = 42).

### 2.2 Algorithms
| Model       | Library / Implementation                                  |
|-------------|------------------------------------------------------------|
| XGBoost     | `xgboost.XGBClassifier`                                    |
| LightGBM    | `lightgbm.LGBMClassifier`                                  |
| SVM         | `sklearn.svm.SVC`                                          |
| TabNet      | `pytorch_tabnet.TabNetClassifier`                          |
| MLP         | `scikeras.KerasClassifier` (TensorFlow backend)            |
| RandomForest| `sklearn.ensemble.RandomForestClassifier`                  |
| Stacking    | `sklearn.ensemble.StackingClassifier` (meta: LogisticRegression) |

### 2.3 Evaluation Metrics
- **Accuracy**: Overall fraction of correct predictions.  
- **Precision**: TP/(TP+FP), critical when false alarms are costly.  
- **Recall**: TP/(TP+FN), critical when missing frauds is unacceptable.  
- **F1-Score**: Harmonic mean of precision & recall.  
- **ROC-AUC**: Discrimination capacity across thresholds.  
- **Time Taken (s)**: Wall‐clock training + batch inference.

## 3 Experimental Results

### 3.1 Numerical Comparison
Table 1 summarizes performance and runtime on the held-out test set.

| Model        | Time (s) | Accuracy | Precision | Recall  | F1-Score | ROC-AUC |
|-------------:|---------:|---------:|----------:|--------:|---------:|--------:|
| XGBoost      |   7.74   | 0.9389   | 0.9335    | 0.9310  | 0.9323   | 0.9382  |
| Stacking     | 311.96   | 0.9387   | 0.9298    | 0.9347  | 0.9323   | 0.9383  |
| LightGBM     |   0.12   | 0.9380   | 0.9302    | 0.9326  | 0.9314   | 0.9375  |
| SVM          |   5.43   | 0.9284   | 0.9196    | 0.9220  | 0.9208   | 0.9278  |
| TabNet       |  89.65   | 0.9109   | 0.9396    | 0.8578  | 0.8968   | 0.9062  |
| MLP          |   2.43   | 0.8573   | 0.8668    | 0.8080  | 0.8364   | 0.8529  |
| RandomForest |   0.05   | 0.9112   | 0.9074    | 0.8921  | 0.8996   | 0.9054  |

*Table 1. Performance metrics and runtime for each classifier.*

### 3.2 Analytical Observations
1. **Accuracy & AUC**: XGBoost and Stacking achieve the highest discrimination (ROC-AUC ≈ 0.938).  
2. **Precision vs. Recall**: TabNet maximizes precision (0.9396) at the cost of recall (0.8578), suiting contexts where false positives must be minimized.  
3. **Efficiency**: LightGBM offers near‐state-of-the-art performance in 0.1 s, ideal for large‐scale or low‐latency applications.  
4. **Ensemble Overhead**: Stacking yields marginal accuracy gains over single models with a ~300 s training penalty.

## 4 Discussion
The results underscore no “one-size-fits-all” solution. Selection criteria depend on:
- **Cost of errors**: If false negatives (missed frauds) are critical, prioritize recall (Stacking, XGBoost).  
- **Operational constraints**: Under strict latency, LightGBM is preferable.  
- **Resource availability**: Deep models (MLP, TabNet) require longer training and may overfit with limited data.

## 5 Conclusion
This comparative study provides a rigorous foundation for model choice in Ethereum fraud detection. Practitioners should balance performance metrics against computation and risk tolerance. Continued refinement—such as threshold tuning and concept drift monitoring—is recommended for production deployment.

## 6 References
1. Chen, T. and Guestrin, C. (2016). “XGBoost: A Scalable Tree Boosting System.” *KDD*.  
2. Ke, G. et al. (2017). “LightGBM: A Highly Efficient Gradient Boosting Decision Tree.” *NIPS*.  
3. Arik, S.Ö. and Pfister, T. (2019). “TabNet: Attentive Interpretable Tabular Learning.” *NeurIPS Workshop*.  
4. Pedregosa, F. et al. (2011). “Scikit‐learn: Machine Learning in Python.” *JMLR*.  
5. Chollet, F. et al. (2015). “Keras.” https://keras.io
