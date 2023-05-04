import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec
from PIL import Image
import cv2


def Compute_Birds_eye_view(dimension, location, theta_ray, alpha , proj_matrix, id):
    # shape = 900
    # fig, ax2 = plt.subplots()

    l = dimension[2] * 15
    h = dimension[0] * 15
    w = dimension[1] * 15
    x = location[0] * 15
    y = location[1] * 15
    z = location[2] * 15

    rot_y = alpha + theta_ray

    R = np.array([[-np.cos(rot_y), np.sin(rot_y)],
                              [np.sin(rot_y), np.cos(rot_y)]])
    t = np.array([x, z]).reshape(1, 2).T

    X_corner1 = [0, l, l, 0]  # -l/2
    Z_corner1 = [w, w, 0, 0]  # -w/2

    X_corner1 += -w / 2
    Z_corner1 += -l / 2

    # bounding box in object coordinate
    corner2D = np.array([X_corner1, Z_corner1])
    # rotate
    corner2D = R.dot(corner2D)
    # translation
    corner2D = t - corner2D
                # in camera coordinate
    corner2D[0] += int(900 / 2)
    corner2D = corner2D.astype(np.int16)
    corner2D = corner2D.T

    pred_corners_2d = np.vstack((corner2D, corner2D[0, :]))

    R = np.array([[np.cos(rot_y), 0, np.sin(rot_y)],
                    [0, 1, 0],
                    [-np.sin(rot_y), 0, np.cos(rot_y)]])

    x_corners = [0, l, l, l, l, 0, 0, 0]  # -l/2
    y_corners = [0, 0, h, h, 0, 0, h, h]  # -h
    z_corners = [0, 0, 0, w, w, w, w, 0]  # -w/2

    x_corners = [i - l / 2 for i in x_corners]
    y_corners = [i - h for i in y_corners]
    z_corners = [i - w / 2 for i in z_corners]

    corners_3D = np.array([x_corners, y_corners, z_corners])
    corners_3D = R.dot(corners_3D)
    corners_3D += np.array([x, y, z]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
    corners_2D = proj_matrix.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]
    corners_2D = corners_2D[:2]

    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]
    bb3d_on_2d_lines_verts = corners_2D[:, bb3d_lines_verts_idx]
    verts = bb3d_on_2d_lines_verts.T
    codes = [Path.LINETO] * verts.shape[0]
    codes[0] = Path.MOVETO
    # codes[-1] = Path.CLOSEPOLYq

    width = corners_2D[:, 3][0] - corners_2D[:, 1][0]
    height = corners_2D[:, 2][1] - corners_2D[:, 1][1]
    # put a mask on the front

    return [pred_corners_2d, id]

def plot_Birds_Eye(birdimg):
    shape = 800
    img = np.full((shape, shape, 3), (255, 255, 255), dtype=np.uint8)
    text_pos_flipped =[]
    for bird, id in birdimg:
        pts = bird.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(img, [pts], (0, 255, 0))

        M = cv2.moments(pts)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        text_pos_flipped.append([str(id), (cx-10, shape - cy-10)])

        x1 = np.linspace(0, shape / 2, num=100)
        x2 = np.linspace(shape / 2, shape, num=100)
        cv2.polylines(img, [np.column_stack((x1, shape / 2 - x1)).astype(int)], False, (0, 0, 255), 1)
        cv2.polylines(img, [np.column_stack((x2, x2 - shape / 2)).astype(int)], False, (0, 0, 255), 1)
        cv2.drawMarker(img, (int(shape / 2), 0), (0, 0, 255), cv2.MARKER_CROSS, 16, 1)
    
    img  = cv2.flip(img, 0)
    for text, pos in text_pos_flipped:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return img