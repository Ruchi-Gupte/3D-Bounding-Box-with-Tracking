## INBUILT YOLO FUNCTIONS
from yolov5.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.datasets import letterbox
import torchvision.transforms as transforms

import os
import time
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import sys
import matplotlib.pyplot as plt

url = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(url, 'yolov5')))
cudnn.benchmark = True

tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((240,320)),
    transforms.Normalize((0.1,0.1,0.1),(0.5,0.5,0.5))
])


class VideoTracker(object):
    def __init__(self):
        self.input_path = 'test.mp4'
        yolo_model_path = 'yolov5/weights/yolov5s.pt'
        self.img_size = 640  
        self.video = cv2.VideoCapture()

        self.device = select_device('')
        self.half = self.device.type != 'cpu'
        use_cuda = self.device.type != 'cpu' and torch.cuda.is_available()

        self.detector = torch.load(yolo_model_path, map_location=self.device)['model'].float()  
        self.detector.to(self.device).eval()
        if self.half:
            self.detector.half() 
        self.names = self.detector.module.names if hasattr(self.detector, 'module') else self.detector.names

        self.printdata = None
        print('Device: ', self.device)


    def __enter__(self):
        self.video.open(self.input_path)
        assert self.video.isOpened()

        os.makedirs('output/', exist_ok=True)
        self.save_video_path = os.path.join('output/', "results.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.save_video_path, fourcc, self.video.get(cv2.CAP_PROP_FPS), (int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        print('Saving output to ', self.save_video_path)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.video.release()
        self.writer.release()
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        frame_no = 0

        while self.video.grab():
            _, frame = self.video.retrieve()                
            bbox, conf = self.object_tracker(frame)   
            if len(conf) == 0:
                print('Frame %d.' % (frame_no), "Objects:  0")   
            else:
                print('Frame %d.' % (frame_no), "Objects: ", len(conf))
                for i,box in enumerate(bbox):
                    x1,y1,x2,y2 = [int(i) for i in box]
                    id = 'car'
                    cv2.rectangle(frame,(x1, y1),(x2,y2),(255,0,0),3)
                    cv2.rectangle(frame,(x1, y1-15),(x1+25,y1), (255,0,0),-1)
                    cv2.putText(frame,id,(x1,y1), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 2)
            self.writer.write(frame)
            frame_no = frame_no + 1

    def object_tracker(self, frame):
        img = letterbox(frame, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float() 
        img = img/255.0  

        if img.ndimension() == 3:
            img = torch.unsqueeze(img,0)

        # ------------------ YOLO -----------------------
        with torch.no_grad():
            pred = self.detector(img, augment=True)[0]  
            pred = non_max_suppression(pred, 0.5, 0.5, classes=[2], agnostic=True)[0]
        if pred is not None and len(pred):  
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            bbox_xywh = pred[:, :4]
            confs = pred[:, 4:5].cpu()
        else:
            bbox_xywh = []
            confs = []

        return bbox_xywh, confs


if __name__ == '__main__':
    with VideoTracker() as runner:
        runner.run()

