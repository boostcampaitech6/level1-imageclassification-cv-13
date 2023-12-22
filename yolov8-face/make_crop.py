import subprocess
import os

train_image_path = '/data/ephemeral/home/yolov8-face/datasets/eval/images'
train_list = sorted(os.listdir(train_image_path))
train_list = [folder for folder in train_list if folder[0] != "."] # 숨김파일 제외
train_list = sorted(train_list)

for i in range(len(train_list)):
    path = os.path.join("datasets/eval/images", train_list[i])
    print(path)
    source_path = "yolo task=detect mode=predict model=yolov8l-face.pt conf=0.25 imgsz=512 line_thickness=1 max_det=1 save_crop=True source="+path
    subprocess.call(source_path, shell=True)
    # break
    # subprocess.call(["yolo task=detect mode=predict model=yolov8l-face.pt conf=0.25 imgsz=512 line_thickness=1 max_det=1 save_crop=True", source_path])
# subprocess.call([""])