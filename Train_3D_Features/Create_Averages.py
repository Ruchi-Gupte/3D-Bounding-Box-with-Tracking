import numpy as np
import os
import json
from tqdm import tqdm

if __name__=='__main__':
    dir_top =  os.path.abspath(os.path.dirname(__file__)) + '/Kitti/training'
    save_path = dir_top + '/class_averages.txt'
    label_path = dir_top + "/label_2/"
    img_path = dir_top + "/image_2/"
    calib_path = dir_top + "/calib/"
    
    ids = [x.split('.')[0] for x in sorted(os.listdir(img_path))]

    class_list = ['Car', 'Van', 'Truck', 'Pedestrian','Person_sitting', 'Cyclist', 'Tram', 'Misc']
    class_count = {}
    class_dict = {}
    for aclass in class_list:
        class_dict[aclass] = 0
        class_count[aclass] = 0

    for i in tqdm(range(len(ids))):
        id = ids[i]
        with open(label_path + '%s.txt'%id) as file:
            for line_num,line in enumerate(file):
                line = line[:-1].split(' ')
                if line[0] in class_dict:
                    boxdim = np.array([line[8], line[9], line[10]]).astype(float)
                    class_dict[line[0]] += boxdim
                    class_count[line[0]] += 1
    
    for key in class_count.keys():
        class_dict[key] = (class_dict[key] / class_count[key]).tolist()
    
    with open('class_averages.json', 'w') as f:
        json.dump(class_dict, f)

