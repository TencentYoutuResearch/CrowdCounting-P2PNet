#!/bin/bash

MODE=$1
DATA_ROOT="/home/pravar/abhinav/data/crowd_counting_datasets/multi_class_agpcc_trolley_and_vehicle/combined_updated"
GPU_ID=2
NUM_CLASSES=2
if [ "$MODE" == "train" ]; then

    python train.py --data_root $DATA_ROOT \
        --dataset_file SHHA \
        --epochs 3500 \
        --lr_drop 3500 \
        --output_dir ./logs \
        --checkpoints_dir ./weights \
        --tensorboard_dir ./logs \
        --lr 0.0001 \
        --lr_backbone 0.00001 \
        --batch_size 16 \
        --eval_freq 1 \
        --gpu_id $GPU_ID \
        --num_classes $NUM_CLASSES \
        --num_workers 0

elif [ "$MODE" == "test" ]; then
    TEST_DATA_PATH="${DATA_ROOT}/test"

    python test.py \
        --img_dir $TEST_DATA_PATH \
        --output_dir ./test_output \
        --weight_path ./weights/best_mae.pth \
        --gpu_id $GPU_ID \
        --num_classes $NUM_CLASSES
else
    echo "Usage: $0 [train|test]"
    exit 1
fi
