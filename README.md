# Project Title
Presence Detection Using CNN

# Getting Started

This repository contains datasets and implementation code for human presence detection using WiFi signals.

These instructions will walk you through how to access the datasets and how to get the project up and running on your local machine.

# Prerequisites
Python interpreter: python3

Development Environment: Ubuntu 16.04 LTS

Packages need to be installed:
json, numpy, matplotlib, keras with tensorflow as backend, 


# Configurations
All the configuration parameters are specified in train_test_conf.py. In the following, we will refer to parameters as conf.param_name.

After cloning the repository, please make two new folders with name conf.data_folder and conf.model_folder
inside this repo:

```
# directory used to store processed data
data_folder = 'data/'
# directory used to store model
model_folder = 'model/'
````

# Datasets
All 16 days' data are available in the following link:
https://drive.google.com/open?id=1t5fnPxIK5dpCVhSNPOnjNcPIWd0COt1U


Ground truth of the dataset is stored in day_conf.json. For example, ground truth of 
the data collected on day 1 is 

```
"day1": {"motion": 6, 
         "mixed_truth": [[0, 1, 1, 0, 1], [1, 0, 0, 0, 1]], 
         "location": "Lab1", 
         "empty": 6, 
         "mixed": 2}, 
         
Explanation: 
On day1 when the experiment was conducted in Lab1, number of runs collected for 
label 0 (empty) and label 1 (motion) is 6 and 6 respecively. Furthermore, two mixture 
runs are collected at the same day with labeling provided in 'mixed_truth'.     
     
```


# How to train
date used for training is specified in train_test_conf.py as 
```
training_date = ['day6', 'day7', 'day8', 'day9', 'day10', 'day11']
```
validation set used for training:
```
training_validate_date = ['day12', 'day13']
```

run the following files sequentially:
1. ```./parse_data_from_log.py -m 1 ```
    
    generate CSI image from the CSI log files
2. ```./data_preprocessing.py -m 1 ```
    
    apply signal processing steps to raw CSI images, prepare input data for CNN
3. ```./data_learing.py -m 1 ```

    obtain a CNN model and save it as conf.model_name:
  
Note: input argument m has value of 1 if in training model otherwise it should be set to be 0.
# How to test
date used for testing the trained model is specified in train_test_conf as
```
test_date = ['day14']
```
1. method I: if the user want to save intermediate data files, please execute the above three 
    steps but each with input argument ```'-m 0'```

2. method II: 
    if the user just needs the detection results without saving intermediate data files, please run 
    ```./wifi_process_combo.py -m 0```

3. method III
    To evaluate the model using mixture runs, please first specify test date in config file

    ```
    draw_date = ['day1', 'day14']
    draw_label = 'mixed'
    ```
    then run ```./combo_no_label.py ```

    Note: if the user want to visualize detection results from other labels, just change conf.draw_label 
    to other values such as 'motion' or 'static'



