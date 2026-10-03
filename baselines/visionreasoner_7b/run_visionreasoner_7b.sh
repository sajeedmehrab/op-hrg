#!/bin/bash

BENCHMARK_ROOT=/path/to/benchmarks
MODEL_PATH=/path/to/VisionReasoner-7B
RESULTS_ROOT=/path/to/results/visionreasoner_7b
BATCH_SIZE=16

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

for dataset in instructpart pascalpart partimagenet; do
    save_dir=${RESULTS_ROOT}/${dataset}
    mkdir -p ${save_dir}
    pids=()
    for chunk_id in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=${chunk_id} conda run --no-capture-output -n visionreasoner python ${SCRIPT_DIR}/infer_visionreasoner_7b.py \
            --dataset ${dataset} \
            --benchmark_root ${BENCHMARK_ROOT} \
            --model_path ${MODEL_PATH} \
            --save_dir ${save_dir} \
            --chunk_id ${chunk_id} \
            --num_chunks 4 \
            --batch_size ${BATCH_SIZE} &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait ${pid} || { echo "ERROR: a GPU chunk failed; stopping."; exit 1; }
    done
    conda run --no-capture-output -n visionreasoner python ${SCRIPT_DIR}/../../benchmarks/aggregate_results.py \
        --save_dir ${save_dir} --dataset ${dataset} --method "VisionReasoner-7B"
done
