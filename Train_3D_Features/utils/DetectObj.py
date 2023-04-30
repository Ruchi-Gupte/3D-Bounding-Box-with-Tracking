import cv2
import numpy as np
import os
from torchvision import transforms
from torch.utils import data
from .File import *
from .ClassAverages import ClassAverages


def generate_bins(bins):
    angle_bins = np.zeros(bins)
    interval = 2 * np.pi / bins
    for i in range(1,bins):
        angle_bins[i] = i * interval
    angle_bins += interval / 2 # center of the bin

    return angle_bins

"""
What is *sorta* the input to the neural net. Will hold the cropped image and
the angle to that image, and (optionally) the label for the object. The idea
is to keep this abstract enough so it can be used in combination with YOLO
"""
class DetectedObject:
    def __init__(self, img, detection_class, box_2d, proj_matrix, label=None):
        # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224)), transforms.Normalize((0.1,0.1,0.1),(0.5,0.5,0.5))])

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
        if isinstance(proj_matrix, str): # filename
            proj_matrix = get_P(proj_matrix)

        self.proj_matrix = proj_matrix
        self.theta_ray = self.calc_theta_ray(img, box_2d, proj_matrix)
        self.img = self.format_img(img, box_2d)
        self.label = label
        self.detection_class = detection_class

    def calc_theta_ray(self, img, box_2d, proj_matrix):
        width = img.shape[1]
        fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
        center = (box_2d[1][0] + box_2d[0][0]) / 2
        dx = center - (width / 2)

        mult = 1
        if dx < 0:
            mult = -1
        dx = abs(dx)
        angle = np.arctan( (2*dx*np.tan(fovx/2)) / width )
        angle = angle * mult

        return angle

    def format_img(self, img, box_2d):        
        xl, yl = box_2d[0]
        xr, yr = box_2d[1]
        batch = self.transform(img[yl:yr+1, xl:xr+1])
        return batch
