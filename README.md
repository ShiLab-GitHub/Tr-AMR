# Tr-AMR
This repository contains the data and Python scripts related to the manuscript titled "Tr-AMR: A Transformer-Based Automatic Modulation Recognition Model with Powerful Temporal Information Extraction Capability". It provides the necessary data sources for model training and evaluation, along with Python scripts that are used for data preprocessing, model implementation and training procedures.
## Requirements
* Python 3.8+
* hdf5 1.14+
* hp5y 3.11+
* torch 2.3.1+
## Dataset
Download the dataset from the following link, and then process it using the scripts sample_snr_data2018.py and sample_modu_data2018.py. Place the processed data in the data folder.
Link: https://www.deepsig.ai/datasets
## File
* train.py: This is the training script for the Tr-AMR model. It contains the necessary code and configurations to train 
the model using the specified dataset and parameters.
* vit_model_2018: This is the code for the model itself.
## Acknowledgements
This study used the publicly available RadioML2018.01A dataset.
