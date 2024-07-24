# AutoEncoder - Small Project

## Overview

This project demonstrates the use of an autoencoder for image colorization. The example uses the ICDAR2015 dataset, which is free to use and can be downloaded from the internet. The project involves data preprocessing, model training, and result evaluation.

## Prerequisites

Ensure you have the following installed:
- Python 3.x
- Required libraries: `numpy`, `matplotlib`, `pickle`, `torch`, `torchvision`, `sklearn`, `PIL`

You can install the necessary libraries using:
```bash
pip install numpy matplotlib pickle-mixin torch torchvision scikit-learn pillow
```
## Step to run the project
1. Download any dataset of your choice and put it into the directory
  
2. Run utils.py to preprocess the images and create a pickle data file.
example command
```bash
python utils.py
```
3. Run train.py to train the model
```bash
python train.py
```
## Result
![image](https://github.com/user-attachments/assets/8fcc93fc-4361-4d78-922b-3432fd203201)
