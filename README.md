# Project Title
Presence Detection Using CNN

## Getting Started

This repository contains datasets and implementation code for human presence detection using WiFi signals.

These instructions will walk you through how to access the datasets and how to get the project up and running on your local machine.

## Prerequisites
Python interpreter: python3

Development Environment: Ubuntu 16.04 LTS

Packages need to be installed:
json, numpy, matplotlib, keras with tensorflow as backend, 


## Datasets and Configurations
put link to datasets here

All configuration parameters are in train_test_conf.py. In the following we will refer to parameters as conf.param_name

After cloning the repository, please make two new folders with name conf.data_folder and conf.model_folder
inside this repo:

```
# folder directory used to store processed data
data_folder = 'data/'
# folder directory used to store model
model_folder = 'model/'
````


## How to train
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
    
    generate CSI image from the log file
2. ```./data_preprocessing.py -m 1 ```
    
    apply signal processing steps to CSI images, prepare data for CNN
3. ```./data_learing.py -m 1 ```

    obtain a CNN model and save it as conf.model_name:
  

## How to test
date used for testing the trained model is specified in train_test_conf as
```
test_date = ['day14']
```
method 1: if the user want to save intermediate data files, please execute the above three 
steps but each with input argument '-m 0'

method 2: if the user just needs the detection results, please run 
```./wifi_process_combo.py -m 0```

date used for evaluating the trained model using mixture datasets
```
draw_date = ['day1', 'day14']
draw_label = 'mixed'
```
run ```./combo_no_label.py ```

Note: if the user want to visualize detection results from other labels, just change conf.draw_label 
to other values such as 'motion' or 'static'



