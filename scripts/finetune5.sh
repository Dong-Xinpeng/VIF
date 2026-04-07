#!/bin/bash
cd /data/dongxinpeng/my_former_work/work


export CUDA_HOME=/inspire/ssd/project/ai4education/public/wxy/paper-download/dxp/cuda-12.5
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export PYTHONPATH=src:$PYTHONPATH

MODEL_PATH='/inspire/ssd/project/ai4education/public/wxy/paper-download/models/mllm'


IMAGE_FOLDER="/data/dongxinpeng/datasets"






ADAPTER_MODE='no_adapter'


GLOBAL_BATCH_SIZE=256
BATCH_PER_DEVICE=16
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

MODEL_NAME="Qwen2.5-VL-7B-Instruct"
NOW_MODE='original'    # 'last_hidden_state_merge' or 'logist_merge' or 'only_former_logist'
TRAIN_FILE_NAME=my_train_data/Cambrian_v4


DATA_PATH="/data/dongxinpeng/datasets/$TRAIN_FILE_NAME.json"
OUTPUT_DIR="checkpoint-output-$ADAPTER_MODE/$MODEL_NAME/$TRAIN_FILE_NAME/$NOW_MODE"

deepspeed /data/dongxinpeng/my_former_work/work/src/train/train_full.py \
    --use_liger True \
    --deepspeed scripts/zero3.json \
    --model_id /data/dongxinpeng/my_former_work/work/checkpoint-output-no_adapter/Qwen2.5-VL-7B-Instruct/ShareGPT4V/my_sharegpt4v_pretrain/original/train_param:+llm+llm_head+vision_tower+merger \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_llm_head False \
    --freeze_merger False \
    --freeze_inf_former True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir  $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((4 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 1e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
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