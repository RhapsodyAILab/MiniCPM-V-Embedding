
# on each node, the script will only run once.
MAX_Q_LEN=$1
MAX_P_LEN=$2
PER_DEV_BATCH_SIZE=$3
POOLING=${4}
ATTENTION=${5}
GPUS_PER_NODE=${6}
BASE_DIR=${7}
SUB_DATASET=${8} # ArguAna, fiqa
MODEL_PATH=${9}


# image fair

# bash eval_mm.sh 512 2048 32 wmean causal 8 /mnt/data/user/tc_agi/xubokai/visualrag_evaldata/ arxivqa /data/checkpoints/multimodal-2024-07-09-061205-model-data-lr-1e-5-softm_temp-0.02-bsz16-ngpus8-nnodes1-inbatch-true-nepoch-1-pooling-wmean-attention-causal-qinstruct-true-cinstruct-false-gradcache-true-passage-stopgrad-false-npassage-1

# bash eval_mm.sh 512 2048 32 wmean causal 8 /mnt/data/user/tc_agi/xubokai/test_data_ocr/ arxivqa /data/checkpoints/multimodal-2024-07-08-190137-model-data-lr-1e-5-softm_temp-0.02-bsz16-ngpus8-nnodes1-inbatch-true-nepoch-1-pooling-wmean-attention-causal-qinstruct-true-cinstruct-false-gradcache-true-passage-stopgrad-false-npassage-1

# bash eval_mm.sh 512 2048 32 wmean causal 8 /mnt/data/user/tc_agi/user/xubokai/formal_data/fair_eval_data_ocrbyminicpmv25_passage_image /mnt/data/user/tc_agi/user/xubokai/formal_data/fair_eval_data_ocrbyminicpmv25_0521_visual_rich_qrels_query 0521_visual_rich,arxivcap_downsampled,arxivqa,docbank_train_poor,docbank_train_rich,docvqa_mp,docvqa_sp,icml_poster,icml_sampled,infographicsqa,nips_sample,visual_anna_poor,visual_anna_rich,visual_embedding_1_visual_poor,visual_embedding_1_visual_rich /data/checkpoints/multimodal-2024-06-09-165843-model-data-lr-1e-5-softm_temp-0.02-bsz8-ngpus8-nnodes2-inbatch-true-nepoch-1-pooling-wmean-attention-causal-qinstruct-true-cinstruct-false-gradcache-true-passage-stopgrad-false-npassage-1


# text fair

# bash eval_mm.sh 512 2048 32 wmean causal 8 /mnt/data/user/tc_agi/user/xubokai/formal_data/fair_eval_data_ocrbyminicpmv25_passage_text /mnt/data/user/tc_agi/user/xubokai/formal_data/fair_eval_data_ocrbyminicpmv25_0521_visual_rich_qrels_query 0521_visual_rich,arxivcap_downsampled,arxivqa,docbank_train_poor,docbank_train_rich,docvqa_mp,docvqa_sp,icml_poster,icml_sampled,infographicsqa,nips_sample,visual_anna_poor,visual_anna_rich,visual_embedding_1_visual_poor,visual_embedding_1_visual_rich /data/checkpoints/multimodal-2024-06-09-165822-model-data-lr-1e-5-softm_temp-0.02-bsz8-ngpus8-nnodes2-inbatch-true-nepoch-1-pooling-wmean-attention-causal-qinstruct-true-cinstruct-false-gradcache-true-passage-stopgrad-false-npassage-1




# MASTER_PORT=23456
CHECKPOINT_DIR="/data/checkpoints"
LOG_DIR="/local/logs"

# 使用 IFS（内部字段分隔符）和 read 命令将字符串分割为数组
IFS=',' read -r -a SUB_DATASET_LIST <<< "$SUB_DATASET"

TIMESTR=$(date "+%Y-%m-%d-%H%M%S")
echo "Time: $TIMESTR"

IDENTITY="multimodal-eval-$TIMESTR-maxq-$MAX_Q_LEN-maxp-$MAX_P_LEN-bsz-$PER_DEV_BATCH_SIZE-pooling-$POOLING-attention-$ATTENTION-gpus-per-node-$GPUS_PER_NODE"
echo "IDENTITY: $IDENTITY"

RESULT_DIR="$CHECKPOINT_DIR/$IDENTITY"

echo "Model Path: $MODEL_PATH"
echo "Data base Path: $BASE_DIR"
echo "Result Dir: $RESULT_DIR"
echo "Log Dir: $LOG_DIR"


pip install transformers==4.40.2
echo "transformers, deepspeed setup succeed!"
pip install -U accelerate
echo "accelerate setup succeed!"
pip install -U datasets
echo "datasets setup succeed!"
cd /local/apps/openmatch


sudo pip uninstall torchvision -y
pip install /mnt/data/user/tc_agi/user/xubokai/torchvision-0.14.1+cu117-cp310-cp310-linux_x86_64.whl
echo "torchvision-cu117 setup succeed!"


cd pytrec_eval
pip install . # here do not use -e .
echo "pytrec_eval setup succeed!"
cd ..


cd pytorch-image-models-0.9.16
pip install . # here do not use -e .
echo "pytorch-image-models-0.9.16 setup succeed!"
cd ..


pip install .

# pip install --upgrade packaging
# pip install --upgrade wheel
# # pip install flash-attn
# pip install -U flash-attn==2.1.0 --no-build-isolation
echo "openmatch setup succeed!"


# LOCAL_DATASET_PATH="/local/apps/openmatch/dataset_tmp"
# copy files
# echo "copying data"
# cp -r $DATASET_PATH $LOCAL_DATASET_PATH
# echo "copied data to $LOCAL_DATASET_PATH"


# step1: distributed parallel encode on one node(shared storage is needed for multiple nodes), multiple GPU encode sharded query -> embedding.query.rank.{process_rank} (single file by default, hack is needed for multiple file)
# step2: distributed parallel encode on one node(shared storage is needed for multiple nodes), multiple GPU encode sharded corpus -> embedding.query.rank.{process_rank}.{begin_id}-{end_id} (multiple by default,hack is needed for single file)
# step3: distributed parallel retrieval on one node (shared storage is needed for multiple nodes), multiple gpu retrieve its part of query, and corpus will share, but load batches by batches (embedding.query.rank.{process_rank}) and save trec file trec.rank.{process_rank}
# step 4: master collect trec file and calculate metrics


for SUB_DATASET in "${SUB_DATASET_LIST[@]}"
do
    echo "Evaluating: $SUB_DATASET"
    # THIS_DATASET_PATH="$DATASET_PATH/$SUB_DATASET"
    
    THIS_RESULT_DIR="$RESULT_DIR/$SUB_DATASET"
    echo "This dataset result dir: $THIS_RESULT_DIR"

    # QUERY_TEMPLATE_PATH="./Eval_Instruction/${SUB_DATASET}.query.txt"
    # QUERY_TEMPLATE=$(cat $QUERY_TEMPLATE_PATH)
    # CORPUS_TEMPLATE="<title> <text>"

    CORPUS_PATH="${BASE_DIR}/${SUB_DATASET}/${SUB_DATASET}-eval-docs.jsonl"
    QUERY_PATH="${BASE_DIR}/${SUB_DATASET}/${SUB_DATASET}-eval-queries.jsonl"
    QRELS_PATH="${BASE_DIR}/${SUB_DATASET}/${SUB_DATASET}-eval-qrels.tsv"

    echo "CORPUS_PATH: $CORPUS_PATH" 
    echo "QUERY_PATH: $QUERY_PATH" 
    echo "QRELS_PATH: $QRELS_PATH" 

    QUERY_TEMPLATE="Represent this query for retrieving relavant documents: <text>"
    CORPUS_TEMPLATE="<text>"
    
    torchrun --nnodes=$WORLD_SIZE \
        --node_rank=$RANK \
        --nproc_per_node=$GPUS_PER_NODE \
        --rdzv_id=1 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        src/openmatch/driver/mm_eval_pipeline.py \
        --qrels_path $QRELS_PATH \
        --query_path $QUERY_PATH \
        --corpus_path $CORPUS_PATH \
        --model_name_or_path "$MODEL_PATH" \
        --output_dir "$THIS_RESULT_DIR" \
        --query_template "$QUERY_TEMPLATE" \
        --doc_template "$CORPUS_TEMPLATE" \
        --q_max_len $MAX_Q_LEN \
        --p_max_len $MAX_P_LEN  \
        --per_device_eval_batch_size $PER_DEV_BATCH_SIZE \
        --dataloader_num_workers 1 \
        --fp16 \
        --use_gpu \
        --overwrite_output_dir false \
        --max_inmem_docs 1000000 \
        --normalize true \
        --pooling "$POOLING" \
        --attention "$ATTENTION" \
        --attn_implementation "flash_attention_2" \
        --phase "encode" \

    torchrun --nnodes=$WORLD_SIZE \
        --node_rank=$RANK \
        --nproc_per_node=$GPUS_PER_NODE \
        --rdzv_id=1 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        src/openmatch/driver/mm_eval_pipeline.py \
        --model_name_or_path "$MODEL_PATH" \
        --qrels_path $QRELS_PATH \
        --query_path $QUERY_PATH \
        --corpus_path $CORPUS_PATH \
        --output_dir "$THIS_RESULT_DIR" \
        --use_gpu \
        --phase "retrieve" \
        --retrieve_depth 10 \

done



