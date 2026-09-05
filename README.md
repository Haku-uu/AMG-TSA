# AMG-TSA

Official implementation of **AMG-TSA: Asymptotic Mixed Graph Convolutional with Trend-Aware Self-Attention Networks for Traffic Flow Forecasting**.

## Overview

AMG-TSA is a spatio-temporal traffic forecasting model designed to capture dynamic spatial correlations and nonlinear temporal dependencies. 

## Datasets

Experiments are conducted on two public traffic datasets:

- **METR-LA**
- **PEMS-BAY**

Please prepare the datasets and corresponding adjacency matrix files before training.

## Requirements

Main dependencies include:

```text
Python 3.x
PyTorch
NumPy
SciPy
Pandas
Matplotlib
````

Please install the required packages according to your local PyTorch and CUDA environment.

## Project Structure

```text
AMG-TSA/
├── model.py
├── engine.py
├── train.py
├── test.py
├── util.py
```

## Training

After preparing the dataset, run:

```bash
python train.py
```

Please modify the dataset path, adjacency matrix path, number of nodes, and related experimental settings in `train.py` when switching between METR-LA and PEMS-BAY.

## Testing

Set the trained checkpoint path in `test.py`, then run:

```bash
python test.py
```

## Citation

If this code is useful for your research, please cite our paper:

```text
Asymptotic Mixed Graph Convolutional with Trend-Aware Self-Attention Networks for Traffic Flow Forecasting
```

## Contact

For questions about the code, please open an issue in this repository.

```
