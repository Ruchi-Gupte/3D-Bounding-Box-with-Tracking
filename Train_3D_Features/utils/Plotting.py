import cv2
import numpy as np
from enum import Enum
import itertools

from .File import *
from .Math import *


# takes in a 3d point and projects it into 2d
def project_3d_pt(pt, proj_P):
    point = np.array(pt)
    point = np.append(point, 1)

    point = np.dot(proj_P, point)

    point = point[:2]/point[2]
    point = point.astype(np.int16)

    return point

def plot_3d_box(img, proj_P, ry, dimension, center, id, box2d, thresh=0):
    R = rotation_matrix(ry)
    corners = create_corners(dimension, location=center, R=R)
    blue = (255,0,0)
    box_3d = []
    for corner in corners:
        point = project_3d_pt(corner, proj_P)
        point[0]= np.clip(point[0], box2d[0][0]-thresh, box2d[1][0]+thresh)
        point[1]= np.clip(point[1], box2d[0][1]-thresh, box2d[1][1]+thresh)
        box_3d.append(point)
    
    cv2.rectangle(img,(box_3d[0][0], box_3d[0][1]),(box_3d[3][0], box_3d[3][1]),blue,2)
    cv2.rectangle(img,(box_3d[6][0], box_3d[6][1]), (box_3d[5][0],box_3d[5][1]),blue,2)
    
    cv2.line(img, (box_3d[6][0], box_3d[6][1]), (box_3d[2][0],box_3d[2][1]), blue, 2)
    cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[3][0],box_3d[3][1]), blue, 2)
    cv2.line(img, (box_3d[4][0], box_3d[4][1]), (box_3d[0][0],box_3d[0][1]), blue, 2)
    cv2.line(img, (box_3d[5][0], box_3d[5][1]), (box_3d[1][0],box_3d[1][1]), blue, 2)

    cv2.rectangle(img,(box_3d[0][0],box_3d[0][1]-15),(box_3d[0][0]+15,box_3d[0][1]), blue,-1)
    cv2.putText(img,id,(box_3d[0][0],box_3d[0][1]), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 2)
    return img

    