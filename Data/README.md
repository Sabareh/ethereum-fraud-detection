# Ethereum Fraud Detection Dataset Directory

This directory should contain the transaction data files used for the fraud detection analysis.

## Expected Data Files

The notebooks are generally configured to use files with names like:
- `ethereum_transactions.csv`
- `synthetic_eth_transactions.csv` (generated automatically by notebooks if no data is found)

## Data Format

If you're adding your own Ethereum transaction dataset, it should ideally include the following columns:

- `address`: Ethereum address involved in the transaction
- `value`: Transaction value in wei or ether
- `gas`: Gas used by the transaction
- `gas_price`: Gas price for the transaction
- `transaction_count`: Number of transactions for this address
- `avg_transaction_value`: Average value of transactions for this address
- `min_transaction_value`: Minimum transaction value for this address
- `max_transaction_value`: Maximum transaction value for this address

Optional column for supervised learning:
- `is_fraud`: Binary label indicating if the transaction is fraudulent (1) or not (0)

## Getting Ethereum Transaction Data

You can obtain Ethereum transaction data from several sources:

1. **Public Datasets**:
   - Google BigQuery Ethereum dataset
   - Ethereum ETL project
   - Kaggle Ethereum datasets

2. **Blockchain Explorers APIs**:
   - Etherscan API
   - Infura
   - Alchemy

3. **Generate Synthetic Data**:
   - The notebooks will generate synthetic data for demonstration if no real data is found
