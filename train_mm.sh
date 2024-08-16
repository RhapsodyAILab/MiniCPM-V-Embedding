# on each node, the script will only run once.
MAX_SEQ_LEN=$1
PER_DEV_BATCH_SIZE=$2
SOFTMAX_TEMPERATURE=$3
EPOCH=$4
QUERY_INSTRUCTION=$5 # bool
CORPUS_INSTRUCTION=$6 # bool
DEEPSPEED=$7 # ds_config.json or ds_config_warmup_decay.json or false
LR=$8
MAPPING=$9 # stream data
POOLING=${10}
ATTENTION=${11} 
NPASSAGE=${12}
GRADCACHE=${13}
GRADCACHE_MICRO=${14}
PASSAGE_STOP_GRAD=${15} # by default it is false
MODEL_PATH=${16}
DATASET_PATH=${17}

CHECKPOINT_DIR=/data/checkpoints
LOG_DIR=/data/tensorboard
GPUS_PER_NODE=8
IN_BATCH=true
LORA=false
LORA_R=32
MAX_Q_LEN=$MAX_SEQ_LEN
MAX_P_LEN=$MAX_SEQ_LEN


# 0703 - minicpmv2.0+human annotated filtered image (8gpu)
# bash train_mm.sh 2048 16 0.02 1 true false ds_config_warmup_decay.json 1e-5 false wmean causal 1 true 1 false /mnt/data/user/tc_agi/user/xubokai/MiniCPM-V-2.0 /mnt/data/user/tc_agi/xubokai/visualrag_traindata/train_data_0704a_image

# 0703 - minicpmv2.0-llm-weight+human annotated filtered image (8gpu)
# bash train_mm.sh 2048 16 0.02 1 true false ds_config_warmup_decay.json 1e-5 false wmean causal 1 true 1 false /mnt/data/user/tc_agi/user/xubokai/MiniCPM-V-2.0-llm-weight /mnt/data/user/tc_agi/xubokai/visualrag_traindata/train_data_0704a_merge

# 0703 - minicpmv2.0-llm-weight+human annotated filtered image (8gpu)
# bash train_mm.sh 2048 16 0.02 1 true false ds_config_warmup_decay.json 1e-5 false wmean causal 1 true 1 false /mnt/data/user/tc_agi/user/xubokai/modeling_siglip /mnt/data/user/tc_agi/xubokai/visualrag_traindata/train_data_0704a_image

# 0703 - minicpmv2.0-llm-weight+human annotated filtered image (8gpu)
# bash train_mm.sh 2048 16 0.02 1 true false ds_config_warmup_decay.json 1e-5 false wmean causal 1 true 1 false /mnt/data/user/tc_agi/user/xubokai/modeling_siglip /mnt/data/user/tc_agi/xubokai/visualrag_traindata/train_data_0704a_image

# 0530 main seed train gpt4o

# bash train_mm.sh 1024 8 0.02 1 true false ds_config_warmup_decay.json 1e-5 false wmean causal 1 true 1 false /mnt/data/user/tc_agi/user/xubokai/MiniCPM-V-2.0 /mnt/data/user/tc_agi/user/xubokai/mmtrain/mmtrain0530

# 0530 main seed train minicpmv2_5llama
# bash train_mm.sh 1024 8 0.02 1 true false ds_config_warmup_decay.json 1e-5 false wmean causal 1 true 2 false /mnt/data/user/tc_agi/user/xubokai/MiniCPM-V-2.0 /mnt/data/user/tc_agi/user/xubokai/mmtrain/mmtrain0530b

# simcse for cjb train (ask for large dropout like 0.3 and larger batch size)
# bash train_mm.sh 1024 8 0.02 1 true false ds_config_warmup_decay.json 1e-5 false lasttoken_simcse causal 1 true 2 false /mnt/data/user/tc_agi/user/xubokai/MiniCPM-V-2.0 /mnt/data/user/tc_agi/user/xubokai/mmtrain/xxxx

# export NCCL_IB_QPS_PER_CONNECTION=8

echo "======== Hostname: ========="
echo "IP Addr: $(hostname -I)"
echo "Hostname: $(hostname)"
echo "RANK: $RANK"
echo "Master addr: $MASTER_ENDPOINT"
TIMESTR=$(date "+%Y-%m-%d-%H%M%S")
echo "Time: $TIMESTR"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "DEEPSPEED=$DEEPSPEED"

echo "======== Hyperparameters: =========="
echo "Learning rate:  $LR"

IDENTITY="multimodal-$TIMESTR-model-data-lr-$LR-softm_temp-$SOFTMAX_TEMPERATURE-bsz$PER_DEV_BATCH_SIZE-ngpus$GPUS_PER_NODE-nnodes$WORLD_SIZE-inbatch-$IN_BATCH-nepoch-$EPOCH-pooling-$POOLING-attention-$ATTENTION-qinstruct-$QUERY_INSTRUCTION-cinstruct-$CORPUS_INSTRUCTION-gradcache-$GRADCACHE-passage-stopgrad-$PASSAGE_STOP_GRAD-npassage-$NPASSAGE"
echo "IDENTITY=$IDENTITY"
EXPORT_DIR="$CHECKPOINT_DIR/$IDENTITY"

echo "======== Arguments: =========="
echo "EXPORT_DIR: $EXPORT_DIR"
echo "Model Path: $MODEL_PATH"
echo "Dataset Path: $DATASET_PATH"
echo "Checkpoint Path: $CHECKPOINT_DIR"
echo "Log Dir: $LOG_DIR"

echo "======== Installing openmatch: =========="

pip install transformers==4.40.2
pip install deepspeed==0.13.2
echo "transformers, deepspeed setup succeed!"

pip install -U accelerate==0.27.0
echo "accelerate setup succeed!"

pip install -U datasets
echo "datasets setup succeed!"

pip install -r /local/apps/openmatch/vision/requirements.txt

cd pytorch-image-models-0.9.16
pip install -e .
cd .. # <- at /local/apps/openmatch
echo "modified timm setup succeed!"

sudo pip uninstall torchvision -y
pip install /mnt/data/user/tc_agi/user/xubokai/torchvision-0.14.1+cu117-cp310-cp310-linux_x86_64.whl
echo "torchvision-cu117 setup succeed!"

cd /local/apps/openmatch
pip install .
echo "openmatch setup succeed!"

echo "======== Train begin: =========="

# no klara-hw
# torchrun \
    # --nnodes=$WORLD_SIZE \
    # --nproc_per_node=$GPUS_PER_NODE \
    # --rdzv_id=1 \
    # --rdzv_backend=c10d \
    # --rdzv_endpoint=$MASTER_ENDPOINT:$MASTER_PORT \

# klara-hw
# torchrun \
#     --nnodes=$WORLD_SIZE \
#     --nproc_per_node=$GPUS_PER_NODE \
#     --node_rank=$RANK \
#     --master_addr=$MASTER_ENDPOINT \
#      --master_port=$MASTER_PORT \


# echo "copying dataset to local.."
# cp $DATASET_PATH /local/
# echo "copy ok"
# DATA_FILE_PATH="/local/$(basename $DATASET_PATH)"
# DATA_FILE_PATH=$DATASET_PATH

torchrun \
    --nnodes=1 \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ENDPOINT:$MASTER_PORT \
    src/openmatch/driver/mm_train_dr.py \
    --overwrite_output_dir \
    --output_dir $EXPORT_DIR \
    --model_name_or_path $MODEL_PATH \
    --do_train  \
    --save_steps 500  \
    --train_dir $DATASET_PATH \
    --bf16 \
    --per_device_train_batch_size $PER_DEV_BATCH_SIZE  \
    --train_n_passages $NPASSAGE  \
    --learning_rate $LR  \
    --q_max_len $MAX_Q_LEN  \
    --p_max_len $MAX_P_LEN  \
    --num_train_epochs $EPOCH  \
    --logging_dir "$LOG_DIR/$IDENTITY" \
    --negatives_x_device \
    --softmax_temperature $SOFTMAX_TEMPERATURE \
    --logging_steps 1 \
    --inbatch_loss $IN_BATCH \
    --lora $LORA \
    --lora_r $LORA_R \
    --gradient_checkpointing true \
    --dataloader_num_workers 1 \
    --save_safetensors false \
    --query_instruction $QUERY_INSTRUCTION \
    --corpus_instruction $CORPUS_INSTRUCTION \
    --use_mapping_dataset $MAPPING \
    --normalize true \
    --pooling $POOLING \
    --attention $ATTENTION \
    --attn_implementation "flash_attention_2" \
    --grad_cache_enable $GRADCACHE \
    --grad_cache_micro_batch_size $GRADCACHE_MICRO \
    --passage_stop_grad $PASSAGE_STOP_GRAD \
    --deepspeed $DEEPSPEED \
    --dataloader_drop_last true \
    # --dtype "bfloat16" \
    # --gradient_accumulation_steps $((PER_DEV_BATCH_SIZE / 32)) \
    # --lr_scheduler_type $lr_scheduler_type \
    # --biaxial_loss $BIAXIAL \
    # --grad_cache true \
    # --seed 42 \
