#!/usr/bin/env python3
import json
from test_date_conf import parse_test_days

# folder directory where all the data is stored
log_folder = 'G://wifi_test//upload_wifi_data//'
# folder directory used to store processed data
data_folder = 'data/'
# folder directory used to store model
model_folder = 'model/'
# json file which stores configuration of every day's collected data
save_json_filename = 'day_conf.json'
total_days = 24
use_exist_json = True
if use_exist_json:
    with open(save_json_filename, 'r') as f:
        day_conf = json.loads(f.read())
    if len(day_conf) == 16:
        print('day_conf was loaded successfully')
else:
    day_conf = parse_test_days(log_folder, total_days)
    to_json = json.dumps(day_conf)
    with open(save_json_filename, 'w') as f:
        f.write(to_json)
    print('json file was saved as '+save_json_filename)	

ntx_max, nrx_max, nsubcarrier_max = 3, 3, 56
ntx, nrx, nsubcarrier = 3, 3, 14

n_timestamps = 128
do_fft = True
fft_shape = (n_timestamps, nsubcarrier)
data_shape_to_nn = (50, nsubcarrier, ntx*nrx, 2)
abs_shape_to_nn = (50, nsubcarrier, ntx*nrx)
phase_shape_to_nn = (50, nsubcarrier, ntx*(nrx-1))
time_offset_ratio = 1.0/20.0
D = 1 # H step size
step_size = 33 # CSI image step size
frame_dur = 10 # milliseconds
skip_time = 5000 # milliseconds
# exclude first and last skip_frames in the current run
skip_frames = skip_time//frame_dur


train_label = {'empty': 0, 'motion':1} # key: types of runs; value: class (0 or 1) it belongs
total_classes = 2
draw_date = ['day24', ]
draw_label = 'living_room'
training_date = ['day9', 'day10', 'day11']
training_validate_date = ['day15']
# make sure validation data and training data come from disjoint days
for d in training_validate_date:
    if d in training_date:
        raise ValueError('validation date {} should not appear in train date'.format(d))
test_date = ['day15']
if int(test_date[0][3:]) >= 20:
    # in Apartment
    test_label = {'empty': 0, 'living_room': 1, 'kitchen': 2, 'bedroomI': 3, 'bedroomII': 4}
else:
    # in LabI or LabII
    test_label = {'empty': 0, 'motion': 1}

epochs = 10
# where to save/store the nn model
model_name = model_folder+'wifi_presence_model.h5'
