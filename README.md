## To run 3D BBOX, run in command line:
```
python Run3D.py
```
Place trained model in Train_3D_Features/trained_models/
Model: https://drive.google.com/file/d/1ZH_5GlL8tfSJCdL6JVyCmx3bhWkZWab3/view?usp=sharing

## Change all Paths to required directories

Default Paths:
```
yolo_model_path = 'yolov5/weights/yolov5s.pt'
model_path =  url + '/Train_3D_Features/trained_models/epoch_10.pkl'
deepsort_model_path = "deep_sort/deep/checkpoint/model_orginal_lr2030.pth"
save_video_path = 'output2.mp4v'
img_path = url + "/eval/2011_09_26/image_02/data/"
calib_file = url + "/eval/2011_09_26/calib_cam_to_cam.txt"
```
