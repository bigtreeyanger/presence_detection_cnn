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
**All the configuration parameters are specified in train_test_conf.py**. In the following, we will refer to parameters as conf.param_name.

1. After cloning the repository, please **make two new folders inside this repo directory** by:
```
mkdir data  # used to store data
mkdir data/training # used to store training data
mkdir data/test # used to store test data
mkdir model  # used to store models
```

2. Assign the absolute path to where data are stored to conf.log_folder, for example:
```
log_folder = '/root/share/upload_wifi_data/'
```

# Datasets
All 24 days' data are available in the following link, please download and unzip them:

https://drive.google.com/open?id=1t5fnPxIK5dpCVhSNPOnjNcPIWd0COt1U

**Note**: data on day 17-19 is not provided since on these three days, we conducted real-time comparsion with a PIR sensor and CSIs were thus not recorded. 

## Composition
day index | location | types of run | 
--- | --- | --- | 
1-3 | LabI | 'empty', 'motion', 'mixed' | 
4-19 | LabII | 'empty', 'motion', 'mixed' |
20-24 | Apartment | 'empty', 'living_room', 'kitchen', 'bedroomI', 'bedroomII'|


Ground truth of the dataset is stored in day_conf.json. For example, ground truth of 
the data collected on day 1 and day 24 are:

```
"day1": {"location": "Lab1",
         "motion": 6, 
         "empty": 6,
         "mixed": 2,
         "mixed_truth": [[0, 1, 1, 0, 1], [1, 0, 0, 0, 1]]}, 
         
Explanation: 
On day1 the experiment was conducted in Lab1, number of runs collected for 
label 0 (empty) and label 1 (motion) is 6 and 6 respecively. Furthermore, two mixture 
runs are collected on the same day with truth labeling provided in 'mixed_truth'.     

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
On day 24, the experiment was conducted in Apartment, and there is one run for human free 
environment. Human motions are categorized according to locations. Number of runs collected for motions
in the living room, kitchen, bedroomI and bedroomII are 2, 2, 2 and 2 respectively. 
There are no mixture runs collected on day 24

```


# How to train
date used for training is specified in train_test_conf.py as 
```
training_date = ['day9', 'day10', 'day11', 'day12', 'day13', 'day14']
```
validation set used in training:
```
training_validate_date = ['day15', 'day16']
```
specify label mapping for training and validation data
```
train_label = {'empty': 0, 'motion': 1}

Explanation: 
key: types of runs (refer to Table under composition)
value: classification class (either 0 or 1) the data belong to; 
```
**Note**: only empty runs map to 0, while all the other types of runs should map to 1

Run the following files sequentially:
```
1. ./parse_data_from_log.py -m Y 
    generate CSI images from the CSI log files

2. ./data_preprocessing.py -m Y 
    apply pre-processing steps to raw CSI images, prepare input data for CNN

3. ./data_learing.py -m Y 
    obtain a CNN model and save it as conf.model_name inside folder './model/'.
```
**Note**: To use the code in training mode,  set the value of input argument m to be 'Y'; To use the code in test mode as given below, set its value to be 'N'. 
All the intermediate data files are saved under './data/training/'

# How to test
date used for testing the trained model is specified in train_test_conf as
```
test_date = ['day14']
```
specify label mapping for test data. By default, test_label include all types of runs available on specified test days; 
```
test_label = {'empty': 0, 'motion': 1} # for LabI or LabII only
or
test_label = {'empty': 0, 'living_room': 1, 'kitchen': 2, 'bedroomI': 3, 'bedroomII': 4} # for Apartment only
or
test_label = {'empty': 0, 'motion': 1, living_room': 2, 'kitchen': 3, 'bedroomI': 4, 'bedroomII': 5} # for labs and Apartment

```
**Note**: Even though this is a binary classification problem, different from train_label, class labeling for test days can go beyond 1 when including data from Apartment. This is 
for the users who want to know the detection accuracy at different locations. 

There are three test methods:

1. method I: <br />
    if the user want to save intermediate data files, please execute the above 
    three steps for training but each with input argument ```'-m N'```.
    All the intermediate data files are saved under './data/test/'


2. method II: <br />
    if the user just needs the detection results without saving intermediate data files, please run 
    ```
    ./wifi_process_combo.py -m N
    ```

3. method III: <br />
    To evaluate the model using mixture runs, please first specify draw date in config file and the label you want to display:

    ```
    draw_date = ['day1', 'day14']
    draw_label = 'mixed'
    ```
    then run 
    ```
    ./combo_no_label.py 
    ```

    Note: if the user wants to visualize detection results from other types of run, just change conf.draw_label to other values such as 'motion', 'empty' or location names in the apartment



