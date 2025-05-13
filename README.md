# Detecting and Classifying Fraudulent Ethereum Accounts Using Machine Learning

## Overview
This project implements a comprehensive system for detecting fraudulent Ethereum accounts using both supervised and unsupervised machine learning techniques. It examines transaction patterns, account behaviors, and network interactions to uncover potential fraud.

## Features
- Transaction pattern analysis
- Account behavior profiling
- Network relationship mapping
- Hybrid ML approach (supervised + unsupervised)
- Real-time detection capabilities
- Performance analytics dashboard
- Unsupervised clustering: OPTICS, K-Means, Hierarchical, GaussianMixture, DBSCAN, Birch, AffinityPropagation

## Technology Stack
- **Programming Language**: Python 3.8+
- **ML Libraries**: scikit-learn, TensorFlow, XGBoost
- **Blockchain Integration**: Web3.py, Etherscan API
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

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

## Project Structure
```
├── datasets/               # Raw and processed data
├── src/
│   ├── data/               # Data collection and preprocessing scripts
│   ├── models/             # Model training and fraud detection scripts
│   └── utils/              # Utility functions and helpers
├── notebooks/              # Jupyter notebooks for exploratory analysis
│   ├── Exploratory.ipynb
│   └── Unsupervised ML Models/  # clustering/anomaly notebooks
│       ├── OPTICS_Unsupervised.ipynb
│       ├── KMeans_Unsupervised.ipynb
│       ├── Hierarchical_Unsupervised.ipynb
│       ├── GaussianMixture_Unsupervised.ipynb
│       ├── DBSCAN_Unsupervised.ipynb
│       ├── Birch_Unsupervised.ipynb
│       └── AffinityPropagation_Unsupervised.ipynb
├── tests/                  # Unit tests for the application
├── ui/                     # Streamlit user interface
│   └── streamlit_app.py
└── docs/                   # Additional documentation and resources
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
streamlit run ui/streamlit_app.py
```

### b) (Legacy) top-level app  
```bash
streamlit run streamlit_app.py
```

6. Unsupervised Analysis:
Open and run any notebook under `notebooks/Unsupervised ML Models/` to explore clustering-based anomaly detection.

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