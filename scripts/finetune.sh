#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH


MODEL_PATH=
IMAGE_FOLDER=
DATA_PATH=
OUTPUT_DIR=

ADAPTER_MODE='self+cross_share_head'
GLOBAL_BATCH_SIZE=256
BATCH_PER_DEVICE=16
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

NOW_MODE='last_hidden_state_merge-with_null_vision_token-for_cycle' 




deepspeed src/train/train_full.py \
    --use_liger True \
    --deepspeed scripts/zero3.json \
    --model_id  $MODEL_PATH \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_llm_head False \
    --freeze_merger False \
    --freeze_inf_former False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir  $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((1 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 1e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --inf_former_lr 2e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 10000  \
    --save_total_limit 10 \
    --dataloader_num_workers 32 \
    --my_decoder_mode $NOW_MODE \
    --adapter_mode $ADAPTER_MODE \
    --save_only_model