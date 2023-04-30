"""
Images must be in ./Kitti/testing/image_2/ and camera matricies in ./Kitti/testing/calib/
Uses YOLO to obtain 2D box, PyTorch to get 3D box, plots both
"""

## INBUILT YOLO FUNCTIONS
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device
from yolov5.utils.datasets import letterbox
import torch.backends.cudnn as cudnn

from Train_3D_Features.utils.DetectObj import *
from Train_3D_Features.utils.Math import *
from Train_3D_Features.utils.Plotting import *
from Train_3D_Features.utils import ClassAverages
from Train_3D_Features import Model
import sys

import os
import time

import numpy as np
import cv2

import torch
import torch.nn as nn
from torchvision.models import vgg


url = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(url, 'yolov5')))
cudnn.benchmark = True

def main():
    yolo_model_path = 'yolov5/weights/yolov5s.pt'
    device = select_device('')

    detector = torch.load(yolo_model_path, map_location=device)['model'].float()  
    detector.to(device).eval()
    names = detector.module.names if hasattr(detector, 'module') else detector.names        

    # load torch
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        exit()
    else:
        print('Using previous model %s'%model_lst[-1])
        my_vgg = vgg.vgg19_bn(pretrained=True)
        model = Model.Model(features=my_vgg.features, bins=2).cuda()

        checkpoint = torch.load(weights_path + '/'+ model_lst[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    averages = ClassAverages.ClassAverages()

    # TODO: clean up how this is done. flag?
    angle_bins = generate_bins(2)

    image_dir = "eval/image_2/"
    cal_dir = "camera_cal/"

    img_path = os.path.abspath(os.path.dirname(__file__)) + "/" + image_dir
    # using P_rect from global calibration file
    calib_file = os.path.abspath(os.path.dirname(__file__)) + "/Train_3D_Features/utils/calib_cam_to_cam.txt"

    # using P from each frame
    # calib_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/testing/calib/'

    try:
        ids = [x.split('.')[0] for x in sorted(os.listdir(img_path))]
    except:
        print("\nError: no images in %s"%img_path)
        exit()
    print(ids)
    for img_id in ids:

        start_time = time.time()

        img_file = img_path + img_id + ".png"

        # P for each frame
        # calib_file = calib_path + id + ".txt"

        truth_img = cv2.imread(img_file)
        img = np.copy(truth_img)
        yolo_img = np.copy(truth_img)

        v5img = letterbox(yolo_img)[0]
        v5img = v5img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        v5img = np.ascontiguousarray(v5img)

        v5img = torch.from_numpy(v5img).to(device)
        v5img = v5img.float() 
        v5img = v5img/255.0  

        if v5img.ndimension() == 3:
            v5img = torch.unsqueeze(v5img,0)

        alldetections = detector(v5img, augment=True)[0]
        detections = non_max_suppression(alldetections, 0.5, 0.5, agnostic=True)[0]
        if detections is None:
            continue
        
        detections[:, :4] = scale_coords(v5img.shape[2:], detections[:, :4], truth_img.shape).round()
        
        for detection in detections:
            detection = detection.cpu().numpy()
            matrix = detection[:4].astype(int).reshape(-1, 2).T
            box2d = [tuple(row) for row in matrix.T]
            detected_class = names[int(detection[-1])]

            if not averages.recognized_class(detected_class):
                continue

            try:
                detectedObject = DetectedObject(img, detected_class, box2d, calib_file)
            except:
                continue

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            box_2d = box2d
            detected_class = detected_class

            input_tensor = torch.zeros([1,3,224,224]).cuda()
            input_tensor[0,:,:,:] = input_img

            [orient, conf, dim] = model(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]

            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi

            #Plot 2D and 3D box
            location, X = calc_location(dim, proj_matrix, box_2d, alpha, theta_ray)
            orient = alpha + theta_ray
            plot_2d_box(truth_img, box_2d)
            plot_3d_box(img, proj_matrix, orient, dim, location) # 3d boxes

        numpy_vertical = np.concatenate((truth_img, img), axis=0)
        cv2.imshow('2D vs 3D detections', numpy_vertical)
        cv2.waitKey(0)
       
if __name__ == '__main__':
    main()
