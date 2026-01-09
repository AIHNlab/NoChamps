import os
import torch
import optuna
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
import random
import numpy as np
from datetime import datetime
from run import init_parser
from optuna.pruners import SuccessiveHalvingPruner
from functools import partial
import json

import shutil

def save_trials_callback(study, trial):
    """Save trial results to CSV after each trial"""
    global last_trial_file
    
    # Create results directory
    os.makedirs('hp_results', exist_ok=True)
    
    # Delete previous file
    if last_trial_file and os.path.exists(last_trial_file):
        os.remove(last_trial_file)
    
    # Save new file
    df = study.trials_dataframe()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    last_trial_file = f'hp_results/trials_{args.model}_{args.model_id}.csv'
    df.to_csv(last_trial_file, index=False)

def select_experiment(args):
    # Select the experiment class based on task_name
    if args.task_name == 'long_term_forecast':
        return Exp_Long_Term_Forecast(args)
    elif args.task_name == 'short_term_forecast':
        return Exp_Short_Term_Forecast(args)
    elif args.task_name == 'imputation':
        return Exp_Imputation(args)
    elif args.task_name == 'anomaly_detection':
        return Exp_Anomaly_Detection(args)
    elif args.task_name == 'classification':
        return Exp_Classification(args)
    else:
        return Exp_Long_Term_Forecast(args)
    
def suggest_params(trial, args, hp_configs):
    """Use Optuna to suggest hyperparameters based on the config file.
       Catch special cases for certain models if needed."""
    # Get the hyperparameter ranges from the config file
    parameters = hp_configs['parameters']
    
    # Override parameters for special cases 
    special_cases = hp_configs['special_cases']
    if args.model in special_cases.keys():
        special_parameters = special_cases[args.model]['parameters']
        for param, special_values in special_parameters.items():
            parameters[param] = special_values                
    
    # Get suggested values by Optuna
    for param, param_kwargs in parameters.items():
        if param_kwargs['type'] == 'categorical':
            value = trial.suggest_categorical(param, **param_kwargs['kwargs'])
        elif param_kwargs['type'] == 'float':
            value = trial.suggest_float(param, **param_kwargs['kwargs'])
        elif param_kwargs['type'] == 'int':
            value = trial.suggest_int(param, **param_kwargs['kwargs'])
        else:
            raise ValueError(f"Unknown parameter type: {param_kwargs['type']} for parameter {param}")
        args.__setattr__(param, value)
        
    # Special settings for ModernTCN due to its unique architecture
    if args.model == "ModernTCN":
        args.num_blocks = torch.ones(args.e_layers, dtype=torch.int32).tolist()
        args.large_size = torch.ones(args.e_layers, dtype=torch.int32).tolist() * 51
        args.small_size = torch.ones(args.e_layers, dtype=torch.int32).tolist() * 5
        args.dims = torch.ones(args.e_layers, dtype=torch.int32).tolist() * args.d_model
        args.dw_dims = torch.ones(args.e_layers, dtype=torch.int32).tolist() * args.d_model  

    # Fix the feedforward dimension to be equal to d_model for the HP search
    args.d_ff = args.d_model
    return args

# Define the objective function
def objective_func(trial, args, hp_configs):
    try:
        # collect suggested hyperparameters
        args = suggest_params(trial, args, hp_configs)

        exp = select_experiment(args)
        setting = f'hp_search_{args.model_id}_{args.model}/trial_{trial.number}'

        print(f"Starting trial {trial.number}")
        exp.train(setting, trial)

        # Validate
        vali_data, vali_loader = exp._get_data(flag='val')
        criterion = exp._select_criterion()
        val_loss = exp.vali(vali_data, vali_loader, criterion)
        print(f"Trial {trial.number} validation loss: {val_loss}")
        
        # Clean up any leftover files/resources
        setting = f'hp_search_{args.model_id}_{args.model}/trial_{trial.number}'
        cleanup_path = os.path.join('./checkpoints/', setting)
        if os.path.exists(cleanup_path):
            shutil.rmtree(cleanup_path)

        return val_loss

    except optuna.exceptions.TrialPruned:
        print(f"Trial {trial.number} pruned")
        raise

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        # Log error details
        with open('hp_results/failed_trials.log', 'a') as f:
            f.write(f"Trial {trial.number} failed:\n")
            f.write(f"Parameters: {trial.params}\n")
            f.write(f"Error: {str(e)}\n\n")
        
        # Clean up any leftover files/resources
        setting = f'hp_search_{args.model_id}_{args.model}/trial_{trial.number}'
        cleanup_path = os.path.join('./checkpoints/', setting)
        if os.path.exists(cleanup_path):
            shutil.rmtree(cleanup_path)
            
        # Return worst possible value to ensure failed trials aren't selected
        return float('inf')

def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    hp_seed = 2021
    test_seed = hp_seed
    set_seed(hp_seed)

    parser = init_parser()
    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() else False

    # Handle multi-GPU setup if needed
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # Ensure 'is_training' is set to True
    args.is_training = 1

    # Global to track last file
    last_trial_file = None
    
    # check if args.path_to_hp_config is provided
    if args.path_to_hp_config is None:
        raise ValueError("Provide a valid path to the hyperparameter config file using --path_to_hp_config")
    
    # load the hp config file
    with open(args.path_to_hp_config, 'r') as f:
        hp_configs = json.load(f)

    # Initialize Optuna study with SuccessiveHalvingPruner
    pruner = SuccessiveHalvingPruner(
        min_resource=hp_configs["min_resource"],          # Minimum number of epochs
        reduction_factor=hp_configs["reduction_factor"],      # Factor to reduce the number of trials
        min_early_stopping_rate=hp_configs["min_early_stopping_rate"]
    )

    study = optuna.create_study(direction='minimize', pruner=pruner)
    
    # Set the objective
    objective = partial(objective_func, args=args, hp_configs=hp_configs)

    # Start the optimization
    study.optimize(objective, n_trials=hp_configs["n_trials"], callbacks=[save_trials_callback])

    # Output the best hyperparameters
    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('  Value:', trial.value)
    print('  Params:')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    # Retrain the model with the best hyperparameters
    for param_name, param_value in trial.params.items():
        setattr(args, param_name, param_value)

    args.d_ff = args.d_model
    args.d_temp = args.d_model
    exp = select_experiment(args)
    for ii in range(args.itr):
        test_seed += 1
        set_seed(test_seed)
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii, test_seed)
        exp.train(setting)

        # Test the model
        exp.test(setting)
