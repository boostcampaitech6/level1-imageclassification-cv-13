# python train.py --data_dir ../../yolov8-face/datasets/train/images \
#     --epochs 100 \
#     --resize 384 384 \
#     --batch_size 64 \
#     --valid_batch_size 64 \
#     --model vit_large_patch32_384 \
#     --optimizer Adam \
#     --name vit_large_patch32_384_Adam_e100

# python train.py --data_dir ../../yolov8-face/datasets/train/images \
#     --epochs 30 \
#     --resize 384 384 \
#     --batch_size 64 \
#     --valid_batch_size 64 \
#     --model vit_large_patch32_384 \
#     --optimizer Adam \
#     --name vit_large_patch32_384_Adam_e100_freeze_add_fclayers

# python inference.py --data_dir ../../yolov8-face/datasets/eval \
#     --resize 384 384 \
#     --batch_size 64 \
#     --model vit_large_patch32_384 \
#     --model_dir ./model/vit_large_patch32_384_Adam_e100_freeze_add_fclayers4 \
#     --output_dir ./output/vit_large_patch32_384_Adam_e100_freeze_add_fclayers4

# python train.py --data_dir ../../yolov8-face/datasets/train/images \
#     --epochs 30 \
#     --resize 384 384 \
#     --batch_size 64 \
#     --valid_batch_size 64 \
#     --augmentation CustomAugmentation \
#     --model vit_large_patch32_384 \
#     --optimizer Adam \
#     --name vit_large_patch32_384_Adam_e100_freeze_add_fclayers_softmax_loss1234_customaug

# python inference.py --data_dir ../../yolov8-face/datasets/eval \
#     --resize 384 384 \
#     --batch_size 64 \
#     --model vit_large_patch32_384 \
#     --model_dir ./model/vit_large_patch32_384_Adam_e100_freeze_add_fclayers_softmax_loss1234_customaug5 \
#     --output_dir ./output/vit_large_patch32_384_Adam_e100_freeze_add_fclayers_softmax_loss1234_customaug5

python inference.py --data_dir ../../yolov8-face/datasets/eval \
    --resize 384 384 \
    --batch_size 64 \
    --model vit_large_patch32_384 \
    --model_dir ./model/vit_large_patch32_384_Adam_e100_freeze_add_fclayers_loss1_loss22 \
    --output_dir ./output/vit_large_patch32_384_Adam_e100_freeze_add_fclayers_loss1_loss22