#!/bin/bash
set -e

WORK_DIR=/path/to/dataset_work_dir
INSTRUCTPART_TRAIN_DIR=/path/to/InstructPart/train1800
GROUNDINGDINO_CONFIG=/path/to/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py
GROUNDINGDINO_WEIGHTS=/path/to/groundingdino_swinb_cogcoor.pth
OUTPUT_DATASET_DIR=/path/to/VisionReasoner_InstructPart_Merged

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p ${WORK_DIR}

conda run --no-capture-output -n ophrg python step1_download_visionreasoner_7k.py \
    --output_dir ${WORK_DIR}/VisionReasoner_multi_object_7k_840

pids=()
for chunk_id in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=${chunk_id} conda run --no-capture-output -n sam3 python step2_sam3_boxes_visionreasoner.py \
        --dataset_dir ${WORK_DIR}/VisionReasoner_multi_object_7k_840 \
        --output_dir ${WORK_DIR}/sam3_boxes_visionreasoner \
        --chunk_id ${chunk_id} &
    pids+=($!)
done
for pid in "${pids[@]}"; do
    wait ${pid} || { echo "ERROR: a GPU chunk failed; stopping."; exit 1; }
done

conda run --no-capture-output -n ophrg python step3_build_visionreasoner_for_qwen3.py \
    --dataset_dir ${WORK_DIR}/VisionReasoner_multi_object_7k_840 \
    --sam3_boxes_dir ${WORK_DIR}/sam3_boxes_visionreasoner \
    --output_dir ${WORK_DIR}/VisionReasoner_for_qwen3

conda run --no-capture-output -n ophrg python step4_instructpart_train_metadata.py \
    --instructpart_train_dir ${INSTRUCTPART_TRAIN_DIR} \
    --order_file instructpart_train_order.txt \
    --output_json ${WORK_DIR}/instructpart_train_metadata.json

pids=()
for chunk_id in 0 1; do
    CUDA_VISIBLE_DEVICES=${chunk_id} conda run --no-capture-output -n sam3 python step5_sam3_boxes_instructpart.py \
        --instructpart_train_dir ${INSTRUCTPART_TRAIN_DIR} \
        --order_file instructpart_train_order.txt \
        --output_dir ${WORK_DIR}/sam3_boxes_instructpart \
        --chunk_id ${chunk_id} &
    pids+=($!)
done
for pid in "${pids[@]}"; do
    wait ${pid} || { echo "ERROR: a GPU chunk failed; stopping."; exit 1; }
done

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n grdino python step6_groundingdino_object_boxes_instructpart.py \
    --instructpart_train_dir ${INSTRUCTPART_TRAIN_DIR} \
    --order_file instructpart_train_order.txt \
    --groundingdino_config ${GROUNDINGDINO_CONFIG} \
    --groundingdino_weights ${GROUNDINGDINO_WEIGHTS} \
    --output_json ${WORK_DIR}/groundingdino_object_boxes_instructpart.json

conda run --no-capture-output -n ophrg python step7_build_instructpart_for_qwen3.py \
    --instructpart_train_dir ${INSTRUCTPART_TRAIN_DIR} \
    --metadata_json ${WORK_DIR}/instructpart_train_metadata.json \
    --sam3_boxes_dir ${WORK_DIR}/sam3_boxes_instructpart \
    --groundingdino_json ${WORK_DIR}/groundingdino_object_boxes_instructpart.json \
    --output_dir ${WORK_DIR}/InstructPart_for_qwen3

conda run --no-capture-output -n ophrg python step8_merge.py \
    --visionreasoner_dataset_dir ${WORK_DIR}/VisionReasoner_for_qwen3 \
    --instructpart_dataset_dir ${WORK_DIR}/InstructPart_for_qwen3 \
    --output_dir ${OUTPUT_DATASET_DIR}
