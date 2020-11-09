#!/usr/bin/env python3
import os
import json


def parse_test_days(direc_prefix, total_days, exclude_days):
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
                    value (int) -- total number of motion tests conducted (valid for LabI or LabII)
                    key (str) -- 'living_room' or 'kitchen' or 'bedroomI' or 'bedroomII'
                    value (int) -- total number of motion tests conducted in different rooms  (valid for Apartment)
                    key (str) -- 'empty'
                    value (int) -- total number of tests conducted when there is nobody inside the environment
                    key (str) -- 'mixed'
                    value (int) -- total number of mixed runs (no mixed runs were conducted in Apartment)
                    key (str) -- 'mixed_truth'
                    value (list) -- each entry of the list is also a list that
                                    contains the ground truth of this mixed run.

    '''

    day_conf = {}
    # mapping data to label
    label = {'empty': 0, 'motion': 1}

    for i in range(1, total_days + 1, 1):
        if i in exclude_days: continue
        day_index = 'day' + str(i)
        d_path = direc_prefix + day_index + '/'
        with open(d_path + 'readme.txt', 'r') as f:
            print('processing day {}'.format(i))
            location, cases, mixed_cnt, mixed_state = None, {}, 0, []
            for l in f:
                m = l.split()
                if len(m) == 0:
                    continue
                if 'Location' in m[0]:
                    location = m[-1]
                elif 'mixed' in m[0]:
                    mixed_cnt += 1
                    idx = int(m[0][-2])
                    mixed_index = 'mixed' + str(idx)
                    status = m[1:]
                    mixed_state.append([])
                    for s in status:
                        if 'empty' in s:
                            mixed_state[-1].append(label['empty'])
                        elif 'motion' in s:
                            mixed_state[-1].append(label['motion'])
                        else:
                            print('undefined status in {}'.format(m[0]))
                else:
                    case_type = m[0][:-1]
                    case_cnt = int(m[-1])
                    cases.update({case_type: case_cnt})

        if location == None or cases == {}:
            raise Exception('invalid info  {} {}'.format(location, cases))

        day_conf[day_index] = {'location': location, 'mixed': mixed_cnt, 'mixed_truth': mixed_state}
        day_conf[day_index].update(cases)
        print(day_conf[day_index])
        print('\n')
        for k, v in day_conf[day_index].items():
            if k == 'location' or k == 'mixed_truth':
                continue
            for j in range(1, v + 1, 1):
                f_name = d_path + k + str(j) + '.data'
                if not os.path.exists(f_name):
                    print("{} doesn't exist !!!!".format(f_name))
    return day_conf


def main():
    total_days = 24
    exclude_days = [17, 18, 19]
    data_folder = '/root/share/upload_wifi_data/'
    day_conf = parse_test_days(data_folder, total_days, exclude_days)
    to_json = json.dumps(day_conf)
    # json filename
    save_json_filename = 'day_conf.json'
    # save day_conf to json file
    with open(save_json_filename, 'w') as f:
        f.write(to_json)
    print('json file was saved as ' + save_json_filename)


if __name__ == "__main__":
    main()
