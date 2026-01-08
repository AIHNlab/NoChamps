import json
import pandas as pd
from collections import defaultdict

from exp.exp_model_efficiency import run_experiment
from run import init_parser
from utils.exp_efficiency_analyser import analyze_efficiency_experiments


def extract_configurations(file_path, task_name):
    # Initialize a defaultdict to group the model_lines by (dataset, model, horizon)
    groups = defaultdict(list)
    
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]
    
    for i in range(0, len(lines), 2):
        model_line = lines[i]
        # Retrieve only lines that begin with the specified task_name
        if not model_line.startswith(task_name):
            continue
        # Remove the task_name prefix
        model_line = model_line[len(task_name)+1:]  # +1 to remove the underscore
        
        # Get the parts of the model_line
        parts = model_line.split('_')
        
        # Extract dataset, horizon, and model from the parts
        dataset = parts[0]
        horizon = parts[1]
        model = parts[2]  # The model name follows the model_id
        
        # Group by (dataset, model, horizon)
        key = (dataset, model, horizon)
        
        # If this dataset-model-horizon config was not stored yet, then add it
        if not key in groups:
            groups[key].append(model_line)
            
    # Create MultiIndex columns
    columns_set = set()
    for (dataset, model, horizon) in groups.keys():
        columns_set.add((f'H{horizon}', model))
    
    # Sort columns by horizon (numeric) and model name
    columns = sorted(columns_set, key=lambda x: (int(x[0][1:]), x[1]))
    multi_col = pd.MultiIndex.from_tuples(columns)
    
    # Collect all unique datasets
    datasets = sorted({key[0] for key in groups.keys()})
    
    # Initialize DataFrame with empty strings
    df = pd.DataFrame(index=datasets, columns=multi_col, dtype='object').fillna('')
    
    # Populate the DataFrame with joined settings
    for (dataset, model, horizon), settings in groups.items():
        col = (f'H{horizon}', model)
        df.at[dataset, col] = '\n'.join(settings)    
    return df


def update_config(args, config_str, dataset):
    parts = config_str.split('_')

    # Set the dataset specific parameters
    with open(args.path_to_dataset_config, 'r') as f:
        ds_configs = json.load(f)
    args.main_cycle = ds_configs[dataset]["main_cycle"]
    args.patch_len = ds_configs[dataset]["main_cycle"]
    args.pstride = ds_configs[dataset]["main_cycle"]
    args.c_out = ds_configs[dataset]["input_dims"]
        
    args.model_id = f"{parts[0]}_{parts[1]}" 
    args.model = parts[2]                     
    args.data = parts[3]                      
    args.features = parts[4][2:]              
    args.seq_len = int(parts[5][2:])          
    args.label_len = int(parts[6][2:])        
    args.pred_len = int(parts[7][2:])         
    args.d_model = int(parts[8][2:])
    args.enc_in = args.c_out
    args.dec_in = args.c_out
    args.n_heads = int(parts[9][2:])         
    args.e_layers = int(parts[10][2:])        
    args.d_layers = int(parts[11][2:])        
    args.d_ff = int(parts[12][2:])            

    args.expand = int(parts[13][6:]) if len(parts) > 14 else 2    # expand2 ->2
    args.d_conv = int(parts[14][2:]) if len(parts) > 15 else 4    # dc4 ->4
    args.factor = int(parts[15][2:]) if len(parts) > 16 else 3    # fc3 ->3
    args.embed = parts[16][2:] if len(parts) > 17 else 'timeF'    # ebtimeF ->timeF
    args.distil = parts[17][2:] if len(parts) > 18 else 'True'    # dtTrue ->True
    
    # Employed TimeMixer params
    args.down_sampling_layers=3
    args.down_sampling_method="avg"
    args.down_sampling_window=2

    return args


def run_efficiency_experiments(config_df, args):
    original_file = args.results_file
    efficiency_file = original_file.replace('.txt', '_efficiency.txt')
    
    for dataset in config_df.index:
        for (horizon, model) in config_df.columns:
            config_strs = config_df.loc[dataset, (horizon, model)].split('\n')
            
            for config_str in config_strs:
                if not config_str: continue
                
                args_config = update_config(args, config_str, dataset)
                for key, value in vars(args_config).items():
                    setattr(args, key, value)
                
                try:
                    metrics_train = run_experiment(args, get_flops_params=False)
                    args.is_training = 0
                    metrics_test = run_experiment(args, get_flops_params=True)
                    args.is_training = 1
                    
                    with open(efficiency_file, 'a') as f:
                        f.write(f"{config_str}\n")
                        
                        # Write training metrics with "_train" suffix
                        for key, value in metrics_train.items():
                            if key not in ['flops', 'params']:
                                f.write(f"{key}_train:{value}, ")
    
                        # Write training metrics with "_test" suffix
                        for key, value in metrics_test.items():
                            if key not in ['flops', 'params']:
                                f.write(f"{key}_test:{value}, ")
                                
                        # Write flops and params from test metrics
                        f.write(f"flops:{metrics_test['flops']}, ")
                        f.write(f"params:{metrics_test['params']}\n\n")
                        
                except Exception as e:
                    print(f"Failed {config_str}: {str(e)}")
    
    return efficiency_file


if __name__ == "__main__":
    
    parser = init_parser()
    parser.add_argument('--path_to_dataset_config', required=True, help='Path to the dataset configuration JSON file')
    args = parser.parse_args()

    config_df = extract_configurations(args.results_file, args.task_name)

    efficiency_file = run_efficiency_experiments(config_df, args)
    print(f"Efficiency results saved to {efficiency_file}")
    
    analysis_file = analyze_efficiency_experiments(args.results_file, efficiency_file, config_df)
    print(f"Analysis results saved to {analysis_file}")
    