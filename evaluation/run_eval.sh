#!/bin/bash

MODEL_PATH=<REPLACE_WITH_YOUR_MODEL_PATH>
BENCHMARK_ROOT=/path/to/benchmarks
RESULTS_ROOT=/path/to/results/ophrg
BATCH_SIZE=32

if [[ "${MODEL_PATH}" == "<REPLACE_WITH_YOUR_MODEL_PATH>" ]]; then
    echo "Set MODEL_PATH at the top of run_eval.sh to a HuggingFace-format model folder."
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for dataset in instructpart partimagenet pascalpart; do
    save_dir=${RESULTS_ROOT}/${dataset}
    mkdir -p ${save_dir}
    pids=()
    for chunk_id in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=${chunk_id} conda run --no-capture-output -n ophrg python ${SCRIPT_DIR}/infer_${dataset}.py \
            --model_path ${MODEL_PATH} \
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
    conda run --no-capture-output -n ophrg python ${SCRIPT_DIR}/../benchmarks/aggregate_results.py \
        --save_dir ${save_dir} --dataset ${dataset} --method "OP-HRG"
done
