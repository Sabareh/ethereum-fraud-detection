# Detecting and Classifying Fraudulent Ethereum Accounts Using Machine Learning

## Overview
This project implements a comprehensive system for detecting fraudulent accounts on the Ethereum blockchain using both supervised and unsupervised machine learning approaches. It analyzes transaction patterns, account behaviors, and network relationships to identify potential fraud with high accuracy.

## Features
- Transaction pattern analysis
- Account behavior profiling
- Network relationship mapping
- Hybrid ML approach (supervised + unsupervised)
- Real-time detection capabilities
- Performance analytics dashboard

## Technology Stack
- **Programming Language**: Python 3.8+
- **ML Libraries**: 
  - scikit-learn
  - TensorFlow
  - XGBoost
- **Blockchain Integration**: 
  - Web3.py
  - Etherscan API
- **Data Processing**: 
  - Pandas
  - NumPy
- **Visualization**: 
  - Matplotlib
  - Seaborn

## Installation

1. Clone the repository
```bash
git clone https://github.com/sabare12/ethereum-fraud-detection.git
cd ethereum-fraud-detection
```

2. Set up virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Configure environment variables
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

## Project Structure
```
├── data/               # Data storage
├── models/            # Trained models
├── notebooks/         # Jupyter notebooks
├── src/              # Source code
├── tests/            # Unit tests
└── docs/             # Documentation
```

## Usage

1. Data Collection
```bash
python src/data/collect_data.py
```

2. Preprocessing
```bash
python src/data/preprocess.py
```

3. Training Models
```bash
python src/models/train.py
```

4. Running Detection
```bash
python src/models/detect.py
```

## Model Performance
- Accuracy: 85%+
- False Positive Rate: <5%
- Detection Speed: <2s per transaction

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Open a Pull Request

## License
MIT License

## Author
Victor Oketch Sabare  
SCT213-C002-0061/2021  
Jomo Kenyatta University of Agriculture and Technology

## Acknowledgments
- Professor Isaac Kega (Project Supervisor)
- JKUAT School of Computing and Information Technology
- Ethereum Developer Community

## Contact
- Email: sabarevictor@gmail.com
- GitHub: https://github.com/sabare12