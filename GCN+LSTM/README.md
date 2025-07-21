# Water Network Head Loss Prediction - GCN+LSTM Model

## Project Overview

This project uses a hybrid architecture of GCN (Graph Convolutional Network) and LSTM (Long Short-Term Memory Network) to predict head loss in water distribution networks. The model leverages historical pressure data from 33 monitoring points to forecast 8 head loss values for the next 12 time steps.

## Environment Setup

### Python Version
- Python 3.8.10


## Data Format

### Input Data
1. **Adjacency Matrix File**
   - 34x34 matrix, first row and column are node indices (0-32)
   - Other elements represent weights between monitoring points

2. **Training Data File**
   - 41 columns: first 33 are pressure data, last 8 are head loss data
   - First row is header, others are time series data (one data point every 5 minutes)

### Data Preprocessing
- Automatic data standardization
- Generate input sequences of 12 time steps
- Predict head loss for the next 12 time steps

## Model Architecture

### GCN Layers (2 layers)
- Input dimension: 1 (pressure value)
- Hidden dimension: 64
- Activation: ReLU
- Dropout: 0.5

### LSTM Layers
- Hidden dimension: 512
- Number of layers: 2
- Dropout: 0.5

### Output Layer
- Fully connected: 128 → 64 → 8
- Output: 8 head loss predictions

## Usage

### 1. Prepare Data
Place the following files in the project root directory

### 2. Run Training

# Run training script
python train.py

### 3. Training Parameters
All data-related and training parameters are now managed in `config.py` for data security and easy configuration. Example parameters:
- Batch size: 288
- Learning rate: 0.0001
- Epochs: 300
- Early stopping patience: 50
- Data split: train 80%, validation 10%, test 10%

## Output Results

### Evaluation Metrics
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)

### Model File
- `best_model.pth`: Best trained model

## Project Structure
```
GCN+LSTM/
├── model.py              # Model definition
├── data_loader.py        # Data loading and preprocessing
├── train.py              # Training script
├── config.py             # All data and training configuration
├── requirements.txt      # Dependency list
├── README.md             # Project description
├── venv38/               # Python 3.8 virtual environment
├──                       # Adjacency matrix
└──                       # Training data
```

