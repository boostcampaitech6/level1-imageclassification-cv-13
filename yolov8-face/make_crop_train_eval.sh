#!/bin/bash

# folders=$(ls -d ./datasets/train/images/*)

# for f in "${folders[@]}"
for f in datasets/train/images/*
do 
    if [ -d $f ]
    then 
        echo "$f"
        yolo task=detect mode=predict model=yolov8l-face.pt conf=0.25 imgsz=512 line_thickness=1 max_det=1 save_crop=True source="$f"
    fi
done