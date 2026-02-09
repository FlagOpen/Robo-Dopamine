#!/bin/bash
export PYTHONPATH=$(pwd)
export WANDB_MODE=disabled

# ======================
# Path Configuration
# ======================
MODEL_PATH="tanhuajie2001/Robo-Dopamine-GRM-3B" # modified here
OUTPUT_DIR="./checkpoints/example_grm_finetune" # modified here
DATASETS=example_grm_finetune                   # modified here

CACHE_DIR="./.cache"
mkdir -p $OUTPUT_DIR

# ======================
# Training Hyperparameters
# ======================
# torchrun --nnodes=$WORLD_SIZE --nproc_per_node=8 --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
torchrun --nproc_per_node=8 --nnodes=1 --master_port=29514 \
    qwenvl/train/train_qwen.py \
    --model_name_or_path $MODEL_PATH \
    --tune_mm_llm True \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --dataset_use $DATASETS \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --bf16 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --mm_projector_lr 1e-5 \
    --vision_tower_lr 5e-7 \
    --optim adamw_torch \
    --model_max_length 32768 \
    --data_flatten False \
    --data_packing False \
    --max_pixels 76800 \
    --min_pixels 12544 \
    --base_interval 2 \
    --video_max_frames 8 \
    --video_min_frames 4 \
    --video_max_frame_pixels 1304576 \
    --video_min_frame_pixels 200704 \
    --num_train_epochs 2 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_steps 200 \
    --save_total_limit 5 \
    --eval_strategy "no" \
    --deepspeed ./scripts/zero3.json \
2>&1 | tee ${OUTPUT_DIR}/train.log