NAME=$1
NUM_GPUS=4
BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
MODEL=$2
DATASET=$3

LR=5e-6
for JUDGE_TYPE in "truth"; do
    MODEL_NAME=${NAME}_${JUDGE_TYPE}_judge
    OUTPUT_DIR=judge/judge_models/${MODEL_NAME}/

    echo "Training ${MODEL_NAME} for ${JUDGE_TYPE} prediction."
    echo "Using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

    export WANDB_NAME=${MODEL_NAME}
    accelerate launch \
        --mixed_precision bf16 \
        --num_machines 1 \
        --num_processes $NUM_GPUS \
        --use_deepspeed \
        --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
        src/finetune.py \
        --model_name_or_path $MODEL \
        --use_flash_attn \
        --use_slow_tokenizer \
        --train_file $DATASET \
        --max_seq_length 256 \
        --preprocessing_num_workers 64 \
        --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
        --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
        --learning_rate 5e-6 \
        --lr_scheduler_type linear \
        --warmup_ratio 0.03 \
        --weight_decay 0. \
        --num_train_epochs 5 \
        --output_dir ${OUTPUT_DIR} \
        --with_tracking \
        --report_to wandb \
        --logging_steps 1
done