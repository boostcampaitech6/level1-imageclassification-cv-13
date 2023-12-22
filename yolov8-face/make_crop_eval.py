# from ultralytics import YOLO
import os
# from distutils.dir_util import copy_tree
import shutil
import natsort

# model = YOLO('yolov8l-face.pt')

eval_image_path = 'datasets/eval/images'
croped_image_path = 'runs/detect/predict2701/crops/face'

# train_list = natsort.natsorted(os.listdir(train_image_path))
# train_list = [folder for folder in train_list if folder[0] != "."] # 숨김파일 제외
# train_list = natsort.natsorted(train_list)
croped_list = natsort.natsorted(os.listdir(croped_image_path))

print(len(croped_list))

# print(train_list[0])
# print(croped_list[0])

for i in range(len(croped_list)):
    # train_folder = os.path.join(train_image_path, train_list[i])
    croped_folder = os.path.join(croped_image_path, croped_list[i])
    # print(train_folder)
    print(croped_list[i])
    shutil.copyfile(croped_folder, os.path.join(eval_image_path, croped_list[i]))
    

# for folder in os.listdir(train_image_path):
#     image_folder = os.path.join(train_image_path, folder)
#     print(image_folder)
#     # results = model(image_folder, conf=0.25, imgsz=256, max_det=1, save_crop=True, save=False)