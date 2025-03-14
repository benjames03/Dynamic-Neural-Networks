# Project: Fault Injection in Big-Little Architecture on CIFAR-10

## Overview
- This project implements and evaluates the use of dynamic neural network models on the CIFAR-10 dataset, focusing specifically on LeNet and ResNet architecture, and their combined use in a Big-Little model.
- It also explores fault injection in LeNet by simulating MAC hardware units and their use in NVDLAs Direct Convolution algorithm.
- This provides information on the reliability of dynamic models and their ability to detect faults in classification.

## Setup & Usage

#### 1. Requirements:
- Python 3.10.12, PyTorch 2.4.1, torchvision 0.19.1, NumPy 1.26.4, Matplotlib 3.7.1, Pandas 2.0.0, seaborn 0.13.2, and torchinfo 1.8.0.
- To install, run the following command: ```pip install -r requirements.txt```
- The CIFAR-10 dataset is automatically downloaded if not present.

#### 2. Training:
- Run the training script with: ```python3 train.py [model_name] [epochs]```
    where ```[model_name]``` can be one of: lenet, resnet9, resnet18, or resnet20.

#### 3. Testing & Fault Injection:
- Evaluate and test individual modules by running: ```python3 test.py```
- Conduct fault injection experiments with: ```python3 injection.py```

#### 4. Classification Analysis & Big-Little Model:
- Run the classification experiments using: ```python3 classify.py```
- This script uses the ```big_little.py``` module to combine model predictions based on a confidence threshold of LeNet.

#### 5. Plotting:
- Visualize fault injection results using: ```python3 plot_fi.py```
- Analyze big-little model data and performance metrics using: ```python3 plot_dnn.py```

## File Descriptions
### train.py
- Implements the training loop for the models.
- Supports multiple model options: LeNet, ResNet9, ResNet18, and ResNet20.
- Loads CIFAR-10 with data augmentation and normalization.
- Saves the trained model’s state after training.

### test.py
- Evaluates model performance on CIFAR-10.
- Provides support for both single-process and multiprocess evaluations.
- Includes functionality to simulate faults and compare predictions.

### resnet.py
- Contains definitions for several ResNet architectures (ResNet9, ResNet18, ResNet20).
- Implements residual blocks and shortcut connections.

### lenet.py
- Implements the classic LeNet architecture.
- Includes a simulated version (SimLeNet) which integrates fault injection for linear and convolutional layers.

### conv.py
- Simulates hardware behavior using MAC arrays.
- Defines SimConv2d for convolution operations with fault injection capability.
- Implements SimLinear that leverages a convolutional approach for linear layers.

### injection.py
- Conducts fault injection experiments on the LeNet model.
- Uses multiprocessing to simulate faults and record effects on model accuracy and prediction margins.
- Writes experiment results to an output file.

### classify.py
- Performs classification experiments using a combination of LeNet and ResNet models.
- Evaluates performance across different confidence thresholds.
- Generates experimental data and a confusion matrix for further analysis.

### big\_little.py
- Implements a “big-little” model that fuses LeNet and ResNet.
- Uses a threshold on prediction confidence to decide when to route a sample through the more complex ResNet.
- Collects statistics on model calls to analyze trade-offs between speed and accuracy.

### plot\_fi.py
- Provides functions to plot fault injection results.
- Generates various plots (error bars, box plots, violin plots, strip plots, histograms) to visualize accuracy and prediction margins across fault levels.

### plot\_dnn.py
- Generates plots for analyzing the “big-little” model.
- Displays graphs for accuracy, loss, execution time, and operations over different thresholds.
- Includes a confusion matrix heatmap.

### metrics.py
- Implements utility functions for computing precision, recall, and F1 score per class from a confusion matrix.
- Contains a function to analyze and print a model's confusion matrix along with its calculated metrics.
- Provides functionality to summarize fault injection tests by comparing base performance (fault-free) against various fault conditions from CSV files.

## License & Acknowledgements
The use of PyTorch and Torchvision for machine learning, NumPy and Pandas for data manipulation, and Matplotlib and Seaborn were crucial to the development of this project.

© 2025 Ben James