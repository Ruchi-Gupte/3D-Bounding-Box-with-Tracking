import cv2
import numpy as np
import os
from torchvision import transforms
from torch.utils import data
from .File import *
from .ClassAverages import ClassAverages

class Dataset(data.Dataset):
    def __init__(self, path, bins=2, overlap=0.1):
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((240,320)), transforms.Normalize((0.1,0.1,0.1),(0.5,0.5,0.5))])
        # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

        self.top_label_path = path + "/label_2/"
        self.top_img_path = path + "/image_2/"
        self.top_calib_path = path + "/calib/"

        self.proj_matrix = get_P(os.path.abspath(os.path.dirname(os.path.dirname(__file__)) + '/utils/calib_cam_to_cam.txt'))

        self.ids = [x.split('.')[0] for x in sorted(os.listdir(self.top_img_path))] # name of file
        self.num_images = len(self.ids)

        # create angle bins
        self.bins = bins
        self.angle_bins = np.zeros(bins)
        self.interval = 2 * np.pi / bins
        for i in range(1,bins):
            self.angle_bins[i] = i * self.interval
        self.angle_bins += self.interval / 2 # center of the bin

        self.overlap = overlap
        # ranges for confidence
        # [(min angle in bin, max angle in bin), ... ]
        self.bin_ranges = []
        for i in range(0,bins):
            self.bin_ranges.append(( (i*self.interval - overlap) % (2*np.pi), \
                                (i*self.interval + self.interval + overlap) % (2*np.pi)) )

        # hold average dimensions
        class_list = ['Car', 'Van', 'Truck', 'Pedestrian','Person_sitting', 'Cyclist', 'Tram', 'Misc']
        self.averages = ClassAverages(class_list)

        self.object_list = self.get_objects(self.ids)

        # pre-fetch all labels
        self.labels = {}
        last_id = ""
        for obj in self.object_list:
            id = obj[0]
            line_num = obj[1]
            label = self.get_label(id, line_num)
            if id != last_id:
                self.labels[id] = {}
                last_id = id

            self.labels[id][str(line_num)] = label


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

    def get_objects(self, ids):
        objects = []
        for id in ids:
            with open(self.top_label_path + '%s.txt'%id) as file:
                for line_num,line in enumerate(file):
                    line = line[:-1].split(' ')
                    obj_class = line[0]
                    if obj_class == "DontCare":
                        continue

                    dimension = np.array([float(line[8]), float(line[9]), float(line[10])], dtype=np.double)
                    self.averages.add_item(obj_class, dimension)

                    objects.append((id, line_num))


        self.averages.dump_to_file()
        return objects


    def get_label(self, id, line_num):
        lines = open(self.top_label_path + '%s.txt'%id).read().splitlines()
        label = self.format_label(lines[line_num])

        return label

    def get_bin(self, angle):

        bin_idxs = []

        def is_between(min, max, angle):
            max = (max - min) if (max - min) > 0 else (max - min) + 2*np.pi
            angle = (angle - min) if (angle - min) > 0 else (angle - min) + 2*np.pi
            return angle < max

        for bin_idx, bin_range in enumerate(self.bin_ranges):
            if is_between(bin_range[0], bin_range[1], angle):
                bin_idxs.append(bin_idx)

        return bin_idxs

    def format_label(self, line):
        line = line[:-1].split(' ')

        Class = line[0]

        for i in range(1, len(line)):
            line[i] = float(line[i])

        Alpha = line[3] # what we will be regressing
        Ry = line[14]
        top_left = (int(round(line[4])), int(round(line[5])))
        bottom_right = (int(round(line[6])), int(round(line[7])))
        Box_2D = [top_left, bottom_right]

        Dimension = np.array([line[8], line[9], line[10]], dtype=np.double) # height, width, length
        # modify for the average
        Dimension -= self.averages.get_item(Class)

        Location = [line[11], line[12], line[13]] # x, y, z
        Location[1] -= Dimension[0] / 2 # bring the KITTI center up to the middle of the object

        Orientation = np.zeros((self.bins, 2))
        Confidence = np.zeros(self.bins)

        # alpha is [-pi..pi], shift it to be [0..2pi]
        angle = Alpha + np.pi

        bin_idxs = self.get_bin(angle)

        for bin_idx in bin_idxs:
            angle_diff = angle - self.angle_bins[bin_idx]

            Orientation[bin_idx,:] = np.array([np.cos(angle_diff), np.sin(angle_diff)])
            Confidence[bin_idx] = 1

        label = {
                'Class': Class,
                'Box_2D': Box_2D,
                'Dimensions': Dimension,
                'Alpha': Alpha,
                'Orientation': Orientation,
                'Confidence': Confidence
                }

        return label

    # will be deprc soon
    def parse_label(self, label_path):
        buf = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line[:-1].split(' ')

                Class = line[0]
                if Class == "DontCare":
                    continue

                for i in range(1, len(line)):
                    line[i] = float(line[i])

                Alpha = line[3] # what we will be regressing
                Ry = line[14]
                top_left = (int(round(line[4])), int(round(line[5])))
                bottom_right = (int(round(line[6])), int(round(line[7])))
                Box_2D = [top_left, bottom_right]

                Dimension = [line[8], line[9], line[10]] # height, width, length
                Location = [line[11], line[12], line[13]] # x, y, z
                Location[1] -= Dimension[0] / 2 # bring the KITTI center up to the middle of the object

                buf.append({
                        'Class': Class,
                        'Box_2D': Box_2D,
                        'Dimensions': Dimension,
                        'Location': Location,
                        'Alpha': Alpha,
                        'Ry': Ry
                    })
        return buf

