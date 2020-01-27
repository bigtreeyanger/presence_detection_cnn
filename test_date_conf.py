#!/usr/bin/python3
import os
import json

def parse_test_days(direc_prefix, total_days):
	'''
	Description:
	generate a dictionary that stores the configurations of each day's collected data
	by parsing readme.txt

	Input:
		direc_prefix (str):
			folder directory where all the data is stored
		total_days (int):
			total number of experimental days

	Output:
		return day_conf

		day_conf (dict): 
			key (str) -- 'day'+str(index) (index starts from 1 to total_days)
			value (dict)-- test_conf (dictionary)

		test_conf (dict): 
			key (str) -- "location"
			value (str) -- where the experiment was conducted
			key (str) -- 'motion'
			value (int) -- total number of motion tests conducted
			key (str) -- 'empty'
			value (int) -- test number of tests conducted when there is nobody inside the lab
			key (str) -- 'mixed'
			value (dict) -- mixed_conf

		mixed_conf (dict): 
			key (str) -- 'mixed'+str(test_index) (test_index starts from 1)
			value (list) -- ground truth of this mixed test
	'''

	day_conf = {}
	# mapping data to label
	label = {'empty': 0, 'motion': 1}

	for i in range(1, total_days+1, 1):
		day_index = 'day'+str(i)
		d_path = direc_prefix+day_index+'/'
		with open(d_path+'readme.txt', 'r') as f:
			print('processing day {}'.format(i))
			location, empty_cnt, motion_cnt, mixed_state = None, -1, -1, {}
			for l in f:
				m = l.split()
				if len(m) == 0:
					continue
				if 'Location' in m[0]:
					location = m[-1]
				if 'motion' in m[0]:
					motion_cnt = int(m[-1])
				if 'empty' in m[0]:
					empty_cnt = int(m[-1])
				if 'mixed' in m[0]:
					idx = int(m[0][-2])
					mixed_index = 'mixed'+str(idx)
					status = m[1:]
					mixed_state[mixed_index] = []
					for s in status:
						if 'empty' in s:
							mixed_state[mixed_index].append(label['empty'])
						elif 'motion' in s:
							mixed_state[mixed_index].append(label['motion'])
						else:
							print('undefined status in {}'.format(m[0]))
			if location == None or empty_cnt == -1 or motion_cnt == -1:
				raise Exception('invalid info  {} {} {}'.format(location, empty_cnt, motion_cnt))

			day_conf[day_index] = {'location': location, 'motion': motion_cnt,
			'empty': empty_cnt, 'mixed': mixed_state}
			print(day_conf[day_index])
			print('\n')
			for k, v in day_conf[day_index].items():
				if k == 'location':
					continue
				if k == 'mixed':
					total_cnt = len(v)
				else:
					total_cnt = v
				for j in range(1, total_cnt+1, 1):
					f_name = d_path+k+str(j)+'.data'
					if not os.path.exists(f_name):
						print("{} doesn't exist".format(f_name))
	return day_conf


def main():
	total_days = 16
	data_folder = 'G:\\wifi_test\\upload_wifi_data\\'
	day_conf = parse_test_days(data_folder, total_days)
	to_json = json.dumps(day_conf)
	# json filename
	save_json_filename = 'day_conf.json'
	# save day_conf to json file
	with open(save_json_filename, 'w') as f:
		f.write(to_json)
	print('json file was saved as '+save_json_filename)	




if __name__ == "__main__":
	main()


