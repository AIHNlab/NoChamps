#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Check if pred_len and script_name arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <pred_len> <script_name> [results_file]"
    echo "Example: $0 96 run.py"
    echo "Example: $0 96 run.py custom_results.txt"
    exit 1
fi

pred_len=$1
script_name=$2
results_file=${3:-"results_default.txt"}

echo "Using prediction length: $pred_len"
echo "Using script: $script_name"
echo "Using results file: $results_file"

model_names=(
    "DLinear"
    "iTransformer"
    "PatchTST"
    "TimeMixer"
    "TimeXer"
    "iPatch"
    "SMamba"
    "xLSTMTime"
    "ModernTCN"
)

# Datasets and their specific settings
declare -A datasets=(
    ["BenzeneConcentration"]="input_dims=8 main_cycle=24 root_path=./dataset/UTSD-full-npy/Environment/BenzeneConcentration stride=100 data_path=random.csv data=UTSD"
    ["MotorImagery"]="input_dims=64 main_cycle=96 root_path=./dataset/UTSD-full-npy/Health/MotorImagery stride=100 data_path=random.csv data=UTSD"
    ["TDBrain"]="input_dims=33 main_cycle=48 root_path=./dataset/UTSD-full-npy/Health/TDBrain_csv stride=100 data_path=random.csv data=UTSD"
    ["BeijingAir"]="input_dims=9 main_cycle=24 root_path=./dataset/UTSD-full-npy/Environment/BeijingPM25Quality stride=100 data_path=random.csv data=UTSD"
    ["Electricity"]="input_dims=50 main_cycle=24 root_path=./dataset/electricity/ stride=1 data_path=electricity.csv data=custom"
    ["Weather"]="input_dims=21 main_cycle=24 root_path=./dataset/weather/ stride=1 data_path=weather.csv data=custom"
    ["ETTh1"]="input_dims=7 main_cycle=24 root_path=./dataset/ETT-small/ stride=1 data_path=ETTh1.csv data=ETTh1"
    ["ETTm1"]="input_dims=7 main_cycle=96 root_path=./dataset/ETT-small/ stride=1 data_path=ETTm1.csv data=custom"
    ["ETTh2"]="input_dims=7 main_cycle=24 root_path=./dataset/ETT-small/ stride=1 data_path=ETTh2.csv data=ETTh1"
    ["ETTm2"]="input_dims=7 main_cycle=96 root_path=./dataset/ETT-small/ stride=1 data_path=ETTm2.csv data=custom"
    ["Exchange"]="input_dims=8 main_cycle=96 root_path=./dataset/exchange_rate/ stride=1 data_path=exchange_rate.csv data=custom"
    ["KDDCup2018"]="input_dims=1 main_cycle=24 root_path=./dataset/UTSD-full-npy/Nature/kdd_cup_2018_dataset_without_missing_values stride=100 data_path=random.csv data=UTSD"
    ["AustraliaRainfall"]="input_dims=3 main_cycle=24 root_path=./dataset/UTSD-full-npy/Environment/AustraliaRainfall stride=100 data_path=random.csv data=UTSD"
    ["PedestrianCounts"]="input_dims=1 main_cycle=24 root_path=./dataset/UTSD-full-npy/Transport/pedestrian_counts_dataset stride=100 data_path=random.csv data=UTSD"
)

# Common settings
seq_len=96 # must be provided but will be searched for in the HP optimization
label_len=24
features="M"
e_layers=2
d_layers=1
factor=3
d_model=16
d_ff=128
itr=3
split=0.8
down_sampling_layers=3
down_sampling_method="avg"
down_sampling_window=2
des="Exp"
path_to_hp_config="./scripts/long_term_forecast/hp_search/hp_configs.json"

# Loop through datasets
for dataset in "${!datasets[@]}"; do
    # Parse dataset-specific settings
    eval ${datasets[$dataset]}
    # Loop through models
    for model in "${model_names[@]}"; do
        python -u $script_name \
            --model_id "${dataset}_${pred_len}" \
            --results_file $results_file \
            --is_training 1 \
            --task_name long_term_forecast \
            --root_path $root_path \
            --data_path $data_path \
            --model $model \
            --data $data \
            --features $features \
            --seq_len $seq_len \
            --label_len $label_len \
            --pred_len $pred_len \
            --e_layers $e_layers \
            --d_layers $d_layers \
            --factor $factor \
            --enc_in $input_dims \
            --dec_in $input_dims \
            --c_out $input_dims \
            --des $des \
            --d_model $d_model \
            --d_ff $d_ff \
            --itr $itr \
            --main_cycle $main_cycle \
            --patch_len $main_cycle \
            --split $split \
            --stride $stride \
            --pstride $main_cycle \
            --down_sampling_layers $down_sampling_layers \
            --down_sampling_method $down_sampling_method \
            --down_sampling_window $down_sampling_window \
            --d_state 16 \
            --path_to_hp_config $path_to_hp_config 
    done
done