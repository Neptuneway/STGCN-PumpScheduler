# LSTM-based Time Series Prediction for Water Network

This project provides an LSTM-based deep learning solution for predicting water head losses using pressure monitoring data. The code is modular, easy to understand, and suitable for research and engineering applications.

## Directory Structure
```
code/LSTM/
├── main.py           # Main program
├── model/
│   └── model_pytorch.py  # PyTorch model implementation
├── data/             # Place your data files here
├── data_config.py    # Data path and column configuration
├── README.md         # This file
```

## Data Configuration
To use your own data, edit `data_config.py`:
- Set `data_file` to your CSV file path
- Set `feature_columns` and `label_columns` to the correct column indices (0-based)

## How to Run
1. Place your data file in the `data/` directory.
2. Edit `data_config.py` as needed.
3. Run the main program

