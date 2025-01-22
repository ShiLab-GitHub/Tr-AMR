# Tr-AMR
This repository contains the data and Python scripts related to the manuscript titled "Tr-AMR: A Transformer-Based Automatic Modulation Recognition Model with Powerful Temporal Information Extraction Capability". It provides the necessary data sources for model training and evaluation, along with Python scripts that are used for data preprocessing, model implementation and training procedures.
## Requirements
* Python 3.8+
* hdf5 1.14+
* hp5y 3.11+
* torch 2.3.1+
## Dataset
You may download the dataset from the following link and process it using the provided scripts sample_snr_data2018.py and sample_modu_data2018.py. Place the processed data into a folder named data.
Download Link: https://www.deepsig.ai/datasets
Alternatively, you can directly download the pre-processed dataset from the link below:
Baidu Cloud Link: https://pan.baidu.com/s/1s9YU2u7BqI0Davfttu-mHQ
Extraction Code: cap6
After downloading, create a folder named data in the current directory and place the dataset inside. If you encounter any issues, please contact the author to obtain the pre-processed dataset.
## File
* train.py: This is the training script for the Tr-AMR model. It contains the necessary code and configurations to train 
the model using the specified dataset and parameters.
* vit_model_2018: This is the code for the model itself.
## Acknowledgements
This study used the publicly available RadioML2018.01A dataset.
