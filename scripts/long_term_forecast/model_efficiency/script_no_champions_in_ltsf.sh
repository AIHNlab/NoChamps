#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

if [ $# -lt 3 ]; then
    echo "Usage: $0 <results_file> <task_name> <path_to_dataset_config> [batch_size] [train_epochs]"
    echo "Example: $0 results.txt long_term_forecast ./scripts/long_term_forecast/model_efficiency/dataset_configs.json"
    echo "Example: $0 results.txt long_term_forecast ./scripts/long_term_forecast/model_efficiency/dataset_configs.json 1 1000"
    exit 1
fi

results_file=$1
task_name=$2
path_to_dataset_config=$3
batch_size=${4:-1}
train_epochs=${5:-1000}

echo "Using results file: $results_file"
echo "Using task name: $task_name"
echo "Using dataset config path: $path_to_dataset_config"
echo "Using batch size: $batch_size"
echo "Using train epochs: $train_epochs"

python -u run_model_efficiency.py \
    --results_file $results_file \
    --task_name $task_name \
    --path_to_dataset_config $path_to_dataset_config \
    --batch_size $batch_size \
    --train_epochs $train_epochs \
    --is_training 1 \
    --model_id test