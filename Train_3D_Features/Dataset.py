import cv2
import os
from torchvision import transforms
from torch.utils import data
import numpy as np
import json

def is_between(min, max, angle):
            max = (max - min) if (max - min) > 0 else (max - min) + 2*np.pi
            angle = (angle - min) if (angle - min) > 0 else (angle - min) + 2*np.pi
            return angle < max

class Dataset(data.Dataset):
    def __init__(self, path, bins=2, overlap=0.1):
        # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((240,320)), transforms.Normalize((0.1,0.1,0.1),(0.5,0.5,0.5))])
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

        self.top_label_path = path + "/label_2/"
        self.top_img_path = path + "/image_2/"
        self.top_calib_path = path + "/calib/"
        self.ids = [x.split('.')[0] for x in sorted(os.listdir(self.top_img_path))]
        self.num_images = len(self.ids)

        with open('class_averages.json', 'r') as f:
            json_str = f.read()
        self.clas_avg = json.loads(json_str)

        # create angle bins
        self.bins = bins
        self.angle_bins = np.zeros(bins)
        self.interval = 2 * np.pi / bins
        for i in range(1,bins):
            self.angle_bins[i] = i * self.interval
        self.angle_bins += self.interval / 2 # center of the bin

        self.overlap = overlap
        self.bin_ranges = []
        for i in range(0,bins):
            self.bin_ranges.append(( (i*self.interval - overlap) % (2*np.pi), (i*self.interval + self.interval + overlap) % (2*np.pi)) )
        
        #Get Labels
        self.object_list = []
        self.labels = {}
        last_id = ""
        for id in self.ids:
            with open(self.top_label_path + '%s.txt'%id) as file:
                for line_num,line in enumerate(file):
                    label = self.get_label(id, line_num)
                    if label==-1:
                        continue
                    if id != last_id:
                        self.labels[id] = {}
                        last_id = id
                    self.labels[id][str(line_num)] = label
                    self.object_list.append((id, line_num))


    def __getitem__(self, index):
        id = self.object_list[index][0]
        line_num = self.object_list[index][1]
        raw_img = cv2.imread(self.top_img_path + '%s.png'%id)
        label = self.labels[id][str(line_num)]
        img = self.format_img(raw_img, label['Box_2D'])
        return img, label

    def __len__(self):
        return len(self.object_list)
    
    def format_img(self, img, box_2d):        
        xl, yl = box_2d[0]
        xr, yr = box_2d[1]
        batch = self.transform(img[yl:yr+1, xl:xr+1])
        return batch
    
    def get_label(self, id, line_num):
        lines = open(self.top_label_path + '%s.txt'%id).read().splitlines()
        line = lines[line_num]
        line = line[:-1].split(' ')
        line = [float(x) if i != 0 else x for i, x in enumerate(line)]
        aclass = line[0]
        if aclass not in self.clas_avg:
            return -1
        alpha = line[3]
        box2d = [(round(line[4]), round(line[5])), (round(line[6]), round(line[7]))]
        dim = np.array([line[8], line[9], line[10]]) - np.array(self.clas_avg[aclass])
        loc = [line[11], line[12], line[13]] 
        loc[1] -= dim[0] / 2 # For KITTI to shift center to the middle

        Orientation = np.zeros((self.bins, 2))
        Confidence = np.zeros(self.bins)

        angle = alpha + np.pi # alpha is [-pi..pi], shift it to be [0..2pi]
        bin_idxs = self.get_bin(angle)

        for bin_idx in bin_idxs:
            angle_diff = angle - self.angle_bins[bin_idx]
            Orientation[bin_idx,:] = np.array([np.cos(angle_diff), np.sin(angle_diff)])
            Confidence[bin_idx] = 1
        label = {'Class': aclass, 'Box_2D': box2d, 'Dimensions': dim, 'Alpha': alpha, 'Orientation': Orientation, 'Confidence': Confidence}
        return label

    def get_bin(self, angle):
        bin_idxs = []
        for bin_idx, bin_range in enumerate(self.bin_ranges):
            if is_between(bin_range[0], bin_range[1], angle):
                bin_idxs.append(bin_idx)
        return bin_idxs