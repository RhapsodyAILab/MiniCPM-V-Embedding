# export q_max_len=64
# export p_max_len=128
# export n_gpus=1
# export port=12345



#### Setup


# cd Library/pytrec_eval
# pip install . # here do not use -e .
# echo "pytrec_eval setup succeed!"
# cd -



# on each node, the script will only run once.
MAX_Q_LEN=1024
MAX_P_LEN=1024

PER_DEV_BATCH_SIZE=32
QUERY_INSTRUCTION=true
CORPUS_INSTRUCTION=false

POOLING="wmean"
ATTENTION="causal"

# CHECKPOINT_MODEL="/mnt/data/user/tc_agi/user/xubokai/MiniCPM-V-2.0"
CHECKPOINT_MODEL="/home/jeeves/MiniCPM-V-2.0"

GPUS_PER_NODE=1

TEMP_DIR="/home/jeeves/tmp-1"

# echo "======== Installing openmatch: =========="
# pip install transformers==4.37.2
# echo "transformers, deepspeed setup succeed!"
# pip install -U accelerate
# echo "accelerate setup succeed!"
# pip install -U datasets
# echo "datasets setup succeed!"

# step1: distributed parallel encode on one node(shared storage is needed for multiple nodes), multiple GPU encode sharded query -> embedding.query.rank.{process_rank} (single file by default, hack is needed for multiple file)
# step2: distributed parallel encode on one node(shared storage is needed for multiple nodes), multiple GPU encode sharded corpus -> embedding.query.rank.{process_rank}.{begin_id}-{end_id} (multiple by default,hack is needed for single file)
# step3: distributed parallel retrieval on one node (shared storage is needed for multiple nodes), multiple gpu retrieve its part of query, and corpus will share, but load batches by batches (embedding.query.rank.{process_rank}) and save trec file trec.rank.{process_rank}
# step 4: master collect trec file and calculate metrics

DATA_BASE="/mnt/data/user/tc_agi/user/xubokai/mmdata"

# modify this
SUB_DATASET="infographicsqa"


CORPUS_PATH="${DATA_BASE}/${SUB_DATASET}_sampled"
echo $CORPUS_PATH

QUERY_PATH="${DATA_BASE}/mm_eval/${SUB_DATASET}-eval-queries.jsonl"
QRELS_PATH="${DATA_BASE}/mm_eval/${SUB_DATASET}-eval-qrels.tsv"

QUERY_TEMPLATE="Represent this query for retrieving relavant documents: <text>"
CORPUS_TEMPLATE="<text>"


# torchrun --nproc_per_node=$GPUS_PER_NODE \
#     src/openmatch/driver/mm_eval_pipeline.py \
#     --qrels_path $QRELS_PATH \
#     --query_path $QUERY_PATH \
#     --corpus_path $CORPUS_PATH \
#     --model_name_or_path "$CHECKPOINT_MODEL" \
#     --output_dir "$TEMP_DIR" \
#     --query_template "$QUERY_TEMPLATE" \
#     --doc_template "$CORPUS_TEMPLATE" \
#     --q_max_len $MAX_Q_LEN \
#     --p_max_len $MAX_P_LEN  \
#     --per_device_eval_batch_size $PER_DEV_BATCH_SIZE \
#     --dataloader_num_workers 1 \
#     --fp16 \
#     --use_gpu \
#     --overwrite_output_dir false \
#     --use_split_search \
#     --max_inmem_docs 1000000 \
#     --normalize true \
#     --pooling "$POOLING" \
#     --attention "$ATTENTION" \
#     --attn_implementation "flash_attention_2" \
#     --phase "encode" \
#     --data_cache_dir "/home/jeeves/cache" \
#     # --attn_implementation "flash_attention_2" \
#     # --data_dir "$DATA_BASE" \


torchrun --nproc_per_node=$GPUS_PER_NODE \
    src/openmatch/driver/mm_eval_pipeline.py \
    --model_name_or_path "$CHECKPOINT_MODEL" \
    --output_dir "$TEMP_DIR" \
    --use_gpu \
    --use_split_search \
    --phase "retrieve" \
    --qrels_path $QRELS_PATH \
    --retrieve_depth 10 \
    
