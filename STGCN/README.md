# STGCN Head Loss Prediction

This project implements a Spatial-Temporal Graph Convolutional Network (STGCN) for predicting head loss in a water distribution network. The model uses data from **33 monitoring points** to predict head loss at **8 target locations**.

## Project Structure

```
Head_loss/
├── config.py             # Configuration file for all parameters
├── main.py               # Main script for training and evaluation
├── README.md             # This documentation file
├── data/                 # Directory for data files
├── model/                # Directory for model architecture
│   ├── layers.py
│   └── models_eval.py
└── script/               # Directory for utility scripts
    ├── dataloader.py
    ├── earlystopping.py
    └── utility.py
```

## Configuration

All key parameters are defined in `config.py`.

### Core Configuration
- `NUM_MONITORING_NODES`: **33** (Number of input monitoring points)
- `NUM_HEAD_LOSS_NODES`: **8** (Number of head loss prediction targets)
- `HISTORICAL_STEPS`: `12` (Number of historical time steps to use as input)
- `PREDICTION_STEPS`: `12` (Number of future time steps to predict)

### Training Configuration
- `BATCH_SIZE`: `288`
- `LEARNING_RATE`: `0.001`
- `EPOCHS`: `300`
- `PATIENCE`: `100` (For early stopping)

## Usage

### Basic Training

To start training the model with the default configuration, simply run:

```bash
python main.py
```

### Custom Training

You can override the default parameters from `config.py` by passing them as command-line arguments:

```bash
python main.py --epochs 500 --lr 0.0005 --batch_size 128
```

### Key Command-Line Arguments
- `--enable_cuda`: Enable CUDA (default: `True`)
- `--seed`: Random seed (default: `42`)
- `--lr`: Learning rate (default: `0.001`)
- `--batch_size`: Batch size (default: `288`)
- `--epochs`: Number of epochs (default: `300`)

## Data Format

### Input Data
The model expects the main data file to be a CSV with:
- **33 columns** for monitoring point data (input features).
- **At least 8 columns** for head loss data (target values), of which the first 8 will be used.
- Each row representing a time step.

### Adjacency Matrix
The graph structure is defined in a separate CSV file, representing the adjacency matrix of the 33 monitoring nodes.

## Output

The training process generates a timestamped directory containing:

- `model.pth`: The trained model weights.
- `training_configs.txt`: A summary of the training configuration.
- `training_loss.csv`: A CSV file with training and validation loss history.
- `training_loss.png`: A plot of the loss curves.
- Evaluation metrics (MSE, MAE, RMSE, WMAPE) printed to the console upon completion. 