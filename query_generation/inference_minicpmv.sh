PROMPT_NAME=$1 # prompt_en_multi.txt
MAX_LENGTH=$2
MODEL_PATH=$3
DATASET_PATH=/home/jeeves/openmatch/document-convert/output.jsonl
export CUDA_VISIBLE_DEVICES=0

# bash vision/inference_minicpmv.sh vlm_ocr_prompt.txt 2048 /mnt/data/user/tc_agi/user/xubokai/MiniCPM-Llama3-V-2_5

BASE_DIR=/home/jeeves/openmatch
BASE_DIR_THIS=$BASE_DIR/vision
CHECKPOINT_DIR=/home/jeeves/caption/checkpoint
LOG_DIR=/home/jeeves/caption/logs

GPUS_PER_NODE=1
PER_DEV_BATCH_SIZE=4
RANK=0
MASTER_ENDPOINT=127.0.0.1
MASTER_PORT=25566
WORLD_SIZE=1

TIMESTR=$(date "+%Y-%m-%d-%H%M%S")
IDENTITY="inference-minicpmv_2_5-$PROMPT_NAME-$TIMESTR"
EXPORT_DIR="$CHECKPOINT_DIR/$IDENTITY"

PROMPT_PATH="$BASE_DIR_THIS/$PROMPT_NAME"

echo "IP Addr: $(hostname -I)"
echo "Hostname: $(hostname)"
echo "RANK: $RANK"
echo "Master addr: $MASTER_ENDPOINT"
echo "Time: $TIMESTR"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "EXPORT_DIR: $EXPORT_DIR"
echo "Model Path: $MODEL_PATH"
echo "Dataset Path: $DATASET_PATH"
echo "Checkpoint Path: $CHECKPOINT_DIR"
echo "Log Dir: $LOG_DIR"

PROMPT="$(cat $PROMPT_PATH)"
echo "prompt: $PROMPT"


cd $BASE_DIR

# install QWen-VL dependency
pip install -r $BASE_DIR_THIS/requirements.txt
echo "QWen-VL dependencies ok."

pip install timm==0.9.10
echo "MiniCPM-V-2.0 dependencies ok."

sudo pip uninstall torchvision -y
pip install /mnt/data/user/tc_agi/user/xubokai/torchvision-0.14.1+cu117-cp310-cp310-linux_x86_64.whl

pip install -U accelerate
echo "accelerate setup succeed!"

pip install -U datasets
echo "datasets setup succeed!"

# pip install transformers==4.37.0
pip install transformers==4.40.2
echo "transformers setup succeed!"

# install openmatch
pip install -e .
echo "openmatch ok."

# launch multiple process with multi-gpus
torchrun \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=$GPUS_PER_NODE \
    --node_rank=$RANK \
    --master_addr=$MASTER_ENDPOINT \
    --master_port=$MASTER_PORT \
    $BASE_DIR_THIS/inference_minicpmv.py \
    --data_path $DATASET_PATH \
    --model_name_or_path $MODEL_PATH \
    --output_dir $EXPORT_DIR \
    --prompt_path $PROMPT_PATH \
    --per_device_eval_batch_size $PER_DEV_BATCH_SIZE \
    --dataloader_num_workers 1 \
    --overwrite_output_dir false \
    --max_new_tokens $MAX_LENGTH \

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
