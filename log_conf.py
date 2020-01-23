import os
direc_prefix = 'G:\\wifi_test\\upload_wifi_data\\'
total_days = 16

day_conf = {}
label = {'empty': 0, 'motion': 1}

for i in range(1, total_days+1, 1):
	d_path = direc_prefix+'day'+str(i)+'\\'
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
				status = m[1:]
				mixed_state[idx] = []
				for s in status:
					if 'empty' in s:
						mixed_state[idx].append(label['empty'])
					elif 'motion' in s:
						mixed_state[idx].append(label['motion'])
					else:
						print('undefined status in {}'.format(m[0]))
		if location == None or empty_cnt == -1 or motion_cnt == -1:
			raise Exception('invalid info  {} {} {}'.format(location, empty_cnt, motion_cnt))


		day_conf[i] = {'location': location, 'motion': motion_cnt,
		'empty': empty_cnt, 'mixed': mixed_state}
		print(day_conf[i])
		print('\n')
		for k, v in day_conf[i].items():
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

