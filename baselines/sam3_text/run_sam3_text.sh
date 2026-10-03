#!/bin/bash

BENCHMARK_ROOT=/path/to/benchmarks
RESULTS_ROOT=/path/to/results/sam3_text
BATCH_SIZE=32

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

for dataset in instructpart pascalpart partimagenet; do
    save_dir=${RESULTS_ROOT}/${dataset}
    mkdir -p ${save_dir}
    pids=()
    for chunk_id in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=${chunk_id} conda run --no-capture-output -n sam3 python ${SCRIPT_DIR}/infer_sam3_text.py \
            --dataset ${dataset} \
            --benchmark_root ${BENCHMARK_ROOT} \
            --save_dir ${save_dir} \
            --chunk_id ${chunk_id} \
            --num_chunks 4 \
            --batch_size ${BATCH_SIZE} &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait ${pid} || { echo "ERROR: a GPU chunk failed; stopping."; exit 1; }
    done
    conda run --no-capture-output -n sam3 python ${SCRIPT_DIR}/../../benchmarks/aggregate_results.py \
        --save_dir ${save_dir} --dataset ${dataset} --method "SAM3 (text prompt)"
done
