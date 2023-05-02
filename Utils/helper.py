import numpy as np
import cv2
from .Math import *

def get_P(cab_f):
    for line in open(cab_f):
        if 'P_rect_02' in line or 'P2:' in line:
            cam_P = line.strip().split(' ')
            cam_P = np.asarray([float(cam_P) for cam_P in cam_P[1:]])
            return_matrix = np.zeros((3,4))
            return_matrix = cam_P.reshape((3,4))
            return return_matrix

    print("Error: File not found at", cab_f)

def get_R0(cab_f):
    for line in open(cab_f):
        if 'R0_rect:' in line or 'R_rect_02:' in line:
            R0 = line.strip().split(' ')
            R0 = np.asarray([float(number) for number in R0[1:]])
            R0 = np.reshape(R0, (3, 3))
            R0_rect = np.zeros([4,4])
            R0_rect[3,3] = 1
            R0_rect[:3,:3] = R0
            return R0_rect

    print("Error: File not found at", cab_f)

def get_tr_to_velo(cab_f):
    for line in open(cab_f):
        if 'Tr_velo_to_cam:' in line or 'T_02:' in line:
            Tr = line.strip().split(' ')
            Tr = np.asarray([float(number) for number in Tr[1:]])
            Tr = np.reshape(Tr, (1, 3))

            Tr_to_velo = np.zeros([4,4])
            Tr_to_velo[3,3] = 1
            Tr_to_velo[:3,3] = Tr
            print(Tr_to_velo)
            return Tr_to_velo

    print("Error: File not found at", cab_f)


# takes in a 3d point and projects it into 2d
def project_3d_pt(pt, proj_P):
    point = np.array(pt)
    point = np.append(point, 1)
    point = np.dot(proj_P, point)
    point = point[:2]/point[2]

    point = point.astype(np.int16)
    return point

def plot_3d_box(img, proj_P, ry, dimension, center, id, box2d=None, thresh=0):

    R = rotation_matrix(ry)
    corners = create_corners(dimension, location=center, R=R)
    color = (255,0,0)
    box_3d = []
    for corner in corners:
        point = project_3d_pt(corner, proj_P)
        if box2d!=None:
            point[0]= np.clip(point[0], box2d[0][0]-thresh, box2d[1][0]+thresh)
            point[1]= np.clip(point[1], box2d[0][1]-thresh, box2d[1][1]+thresh)
        box_3d.append(point)
    
    cv2.rectangle(img,(box_3d[0][0], box_3d[0][1]),(box_3d[3][0], box_3d[3][1]),color,2)
    cv2.rectangle(img,(box_3d[6][0], box_3d[6][1]), (box_3d[5][0],box_3d[5][1]),color,2)
    
    cv2.line(img, (box_3d[6][0], box_3d[6][1]), (box_3d[2][0],box_3d[2][1]), color, 2)
    cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[3][0],box_3d[3][1]), color, 2)
    cv2.line(img, (box_3d[4][0], box_3d[4][1]), (box_3d[0][0],box_3d[0][1]), color, 2)
    cv2.line(img, (box_3d[5][0], box_3d[5][1]), (box_3d[1][0],box_3d[1][1]), color, 2)

    cv2.rectangle(img,(box_3d[0][0],box_3d[0][1]-15),(box_3d[0][0]+15,box_3d[0][1]), color,-1)
    cv2.putText(img,id,(box_3d[0][0],box_3d[0][1]), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 2)
    return img

def generate_bins(bins):
    angle_bins = np.zeros(bins)
    interval = 2 * np.pi / bins
    for i in range(1,bins):
        angle_bins[i] = i * interval
    angle_bins += interval / 2 # center of the bin

    return angle_bins

def plot_2d_box(img, box_2d, id):
    # create a square from the corners
    x1, y1 = box_2d[0]
    x2,y2 = box_2d[1]
   
    # plot the 2d box
    cv2.rectangle(img,(x1, y1),(x2,y2),(255,0,0),3)
    cv2.rectangle(img,(x1, y1+15),(x1+25,y1), (255,0,0),-1)
    cv2.putText(img,id,(x1,y1+15), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 2)

def calc_theta_ray(img, box_2d, proj_matrix):
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