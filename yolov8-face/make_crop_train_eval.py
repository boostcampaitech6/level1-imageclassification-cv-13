# from ultralytics import YOLO
import os
from distutils.dir_util import copy_tree
import natsort

# model = YOLO('yolov8l-face.pt')

train_image_path = '/data/ephemeral/home/yolov8-face/datasets/train/images'
croped_image_path = '/data/ephemeral/home/yolov8-face/runs/detect'

train_list = natsort.natsorted(os.listdir(train_image_path))
train_list = [folder for folder in train_list if folder[0] != "."] # 숨김파일 제외
train_list = natsort.natsorted(train_list)
croped_list = natsort.natsorted(os.listdir(croped_image_path))

print(len(train_list))

print(train_list[0])
print(croped_list[0])

for i in range(len(train_list)):
    train_folder = os.path.join(train_image_path, train_list[i])
    croped_folder = os.path.join(croped_image_path, croped_list[i]+'/crops/face')
    print(train_folder)
    print(croped_folder)
    copy_tree(croped_folder, train_folder)
    

# for folder in os.listdir(train_image_path):
#     image_folder = os.path.join(train_image_path, folder)
#     print(image_folder)
#     # results = model(image_folder, conf=0.25, imgsz=256, max_det=1, save_crop=True, save=False)