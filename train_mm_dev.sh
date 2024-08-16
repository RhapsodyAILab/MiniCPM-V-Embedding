WORLD_SIZE=1
RANK=0
GPUS_PER_NODE=1
MASTER_ENDPOINT=localhost
MASTER_PORT=23456
PER_DEV_BATCH_SIZE=32 # full-parameter, cpmd 2p4b

CHECKPOINT_DIR=/home/jeeves/checkpoints
MODEL_NAME="mm-test-train-stopgrad-bsz$PER_DEV_BATCH_SIZE"
MODEL_PATH=/home/jeeves/MiniCPM-V-2.0
# MODEL_PATH=/home/jeeves/cpm_d-2b_with_pad_token

# DATASET_PATH=/home/jeeves/medi-data-jsonl/train.jsonl
# DATASET_PATH=/home/jeeves/arxivcap-train-v1-jsonl/data.jsonl
DATASET_PATH=/home/jeeves/visual_embedding_1_train_jsonl/data.jsonl
# DATASET_PATH=/home/jeeves/medi-data-jsonl/mm_train_batch_256.jsonl
DATASET_PATH=/home/jeeves/arxivcap-2000-train-jsonl-merged/data.jsonl
LOG_DIR=/home/jeeves/logs

# LR=5e-6
LR=1e-5
# LR=1e-4
SOFTMAX_TEMPERATURE=0.02

# PER_DEV_BATCH_SIZE=8 # lora, cpmd 2p4b
# LORA=true
LORA=false
LORA_R=8
IN_BATCH=true
# IN_BATCH=false
# https://huggingface.co/google/siglip-large-patch16-256

TIMESTR=$(date "+%Y-%m-%d-%H%M%S")\

TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ENDPOINT \
    --master_port=$MASTER_PORT \
    src/openmatch/driver/mm_train_dr.py \
    --overwrite_output_dir \
    --output_dir "$CHECKPOINT_DIR/$MODEL_NAME" \
    --model_name_or_path $MODEL_PATH \
    --do_train  \
    --save_steps 100  \
    --train_path $DATASET_PATH \
    --bf16  \
    --per_device_train_batch_size $PER_DEV_BATCH_SIZE  \
    --train_n_passages 1  \
    --learning_rate $LR  \
    --num_train_epochs 1  \
    --logging_dir "$LOG_DIR/$TIMESTR-$LR-bsz-$PER_DEV_BATCH_SIZE-temp$SOFTMAX_TEMPERATURE" \
    --logging_steps 1 \
    --softmax_temperature $SOFTMAX_TEMPERATURE \
    --negatives_x_device \
    --inbatch_loss $IN_BATCH \
    --gradient_checkpointing true \
    --dataloader_num_workers 1 \
    --use_mapping_dataset false \
    --save_safetensors false \
    --query_instruction true \
    --corpus_instruction true \
    --normalize true \
    --attn_implementation "flash_attention_2" \
    --pooling "drop_wmean" \
    --attention "bidirectional" \
    --q_max_len 1024  \
    --p_max_len 1024  \
    --passage_stop_grad false \
    --grad_cache_enable true \
    --grad_cache_micro_batch_size 2 \
    # --lr_scheduler_type "constant" \
    # --deepspeed "ds_config_warmup_decay.json" \
    # --seed 42
    # --max_steps 100 \
    # --lora $LORA \
    # --lora_r $LORA_R \
