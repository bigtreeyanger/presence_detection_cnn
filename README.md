# Project Title
Presence Detection Using CNN

# Getting Started

This repository contains datasets and implementation code for the paper:

"Harvesting Ambient RF for Presence Detection Through Deep Learning." Yang Liu, Tiexing Wang, Yuexin Jiang and Biao Chen. arXiv link: https://arxiv.org/abs/2002.05770

Please cite this paper if you use the code/data in this repository as part of a published research project.

These instructions will walk you through how to access the datasets and how to get the project up and running on your local machine.

# Prerequisites
Python interpreter: python3.6

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
```

# Datasets
All 24 days' data are available in the following link:

https://drive.google.com/open?id=1t5fnPxIK5dpCVhSNPOnjNcPIWd0COt1U

**Note**: data on day 17-19 is not provided since on these three days, we conducted real-time comparsion with a PIR sensor and CSIs were thus not recorded. 

## Composition
* day 1-3: in LabI
* day 4-19: in LabII
* day 20-24: in Apartment


Ground truth of the dataset is stored in day_conf.json. For example, ground truth of 
the data collected on day 1 and day 24 are:

```
"day1": {"location": "Lab1",
         "motion": 6, 
         "empty": 6,
         "mixed": 2,
         "mixed_truth": [[0, 1, 1, 0, 1], [1, 0, 0, 0, 1]]}, 
         
Explanation: 
On day1 the experiment happened in Lab1, number of runs collected for 
label 0 (empty) and label 1 (motion) is 6 and 6 respecively. Furthermore, two mixture 
runs are collected at the same day with labeling provided in 'mixed_truth'.     

mixed data runs: a continuous run where two states alternates. In other runs, 
only one state is involved, that is CSI images are either labeled as 0 or 1.  
     
```
**Note: mixture runs are for evaluation purpose only, don't include them for training or test set**. 

```
"day24": {"location": "Apartment",
          "mixed": 0, 
          "mixed_truth": [], 
          "empty": 1, 
          "living_room": 2, 
          "kitchen": 2, 
          "bedroomI": 2, 
          "bedroomII": 2}

Explanation:
On day 24, the experiment was conducted in Apartment, and there is one run for empty 
environment. Human motions are categorized according to locations. Number of runs collected for motions
in the living room, kitchen, bedroomI and bedroomII are 2, 2, 2 and 2 respectively. 

```


# How to train
In the repo directory, generate folder **./data/training/** for storing training data if it doesn't exist.<br/>
generate folder **./model/** for storing models. <br/>
date used for training is specified in train_test_conf.py as 
```
training_date = ['day9', 'day10', 'day11', 'day12', 'day13', 'day14']
```
validation set used for training:
```
training_validate_date = ['day15', 'day16']
```
specify label mapping for training and validation data
```
train_label = {'empty': 0, 'motion': 1}

Explanation: 
key: types of runs (for LabI and LabII, it can be either 'empty' or 'motion'; for Apartment, it can be either 'empty' 
or the location where the motions took place, that is 'kitchen', 'living_room','bedroomI' or 'bedroomII')
value: classification class (either 0 or 1) the data belongs to;
```

Run the following files sequentially:
```
1. ./parse_data_from_log.py -m Y 
    generate CSI image from the CSI log files

2. ./data_preprocessing.py -m Y 
    apply signal processing steps to raw CSI images, prepare input data for CNN

3. ./data_learing.py -m Y 
    obtain a CNN model and save it as conf.model_name:
```
**Note**: To use the code in training mode,  set the value of input argument m to be 'Y'; To use the code in test mode, set its value to be 'N' .

# How to test
In the repo directory, generate folder **./data/test/** for storing training data if it doesn't exist.<br/>
date used for testing the trained model is specified in train_test_conf as
```
test_date = ['day14']
```
specify label mapping for test data
```
test_label = {'empty': 0, 'motion': 1} # for LabI or LabII

test_label = {'empty': 0, 'living_room': 1, 'kitchen': 2, 'bedroomI': 3, 'bedroomII': 4} # for Apartment

Explanation: 
similar to train_label. By default, test_label include all types of run on this test day; 
```
**Note**: Even though this is a binary classification problem, class labeling for test day can go beyond 1 in the case of Apartment. This is 
for the users who want to know the detecting accuracy in different locations. 

There are three test methods:

1. method I: if the user want to save intermediate data files, please execute the above three 
    steps but each with input argument ```'-m N'```

2. method II: 
    if the user just needs the detection results without saving intermediate data files, please run 
    ```./wifi_process_combo.py -m N```

3. method III
    To evaluate the model using mixture runs, please first specify test date in config file

    ```
    draw_date = ['day1', 'day14']
    draw_label = 'mixed'
    ```
    then run ```./combo_no_label.py ```

    Note: if the user want to visualize detection results from other labels, just change conf.draw_label 
    to other values such as 'motion' or 'static'



