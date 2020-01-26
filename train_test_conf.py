import json
from test_date_conf import parse_test_days

# folder directory where all the data is stored
log_folder = '/root/media/wifi_test/upload_wifi_data/'
# folder directory used to store processed data
data_folder = 'data/'
# folder directory used to store model
model_folder = 'model/'
# json file which stores configuration of every day's collected data
save_json_filename = 'day_conf.json'
total_days = 16
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
time_offset_ratio = 1.0/20.0
D = 1 # H step size
step_size = 33 # CSI image step size
frame_dur = 10 # milliseconds
skip_time = 5000 # milliseconds
# exclude first and last skip_frames in the current run
skip_frames = skip_time//skip_frames

label = {'empty': 0, 'motion': 1}
total_classes = 2
draw_date = ['day1', 'day2']
draw_label = 'mixed'
training_date = ['day6', 'day7', 'day8', 'day9', 'day10', 'day11']
training_validate_date = ['day12', 'day13']
# make sure validation data and training data come from disjoint days
for d in training_validate_date:
	if d in training_date:
		raise ValueError('validation date {} should not appear in train date'.format(d))
test_date = ['day1']

epoch = 10
# where to save/store the nn model
model_name = model_folder+'wifi_presence_model.h5'