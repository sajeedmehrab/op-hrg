#!/bin/bash

MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct
DATASET_PATH=/path/to/VisionReasoner_InstructPart_Merged
CHECKPOINT_DIR=/path/to/checkpoints/ophrg_qwen3vl4b_instruct
export CUDA_VISIBLE_DEVICES=0,1,2,3

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${SCRIPT_DIR}"
cd "${SCRIPT_DIR}"

# data.val_files is a placeholder: the trainer requires a validation set, but validation is disabled (trainer.val_freq: -1 in config_ophrg.yaml).
python3 -m verl.trainer.main \
    config=./config_ophrg.yaml \
    data.train_files=${DATASET_PATH} \
    data.val_files=${DATASET_PATH} \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.save_checkpoint_path=${CHECKPOINT_DIR}
