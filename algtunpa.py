# -*- coding: utf-8 -*-


import gymnasium as gym
import os
import argparse  
import json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
import cloudpickle
from sb3_contrib import RecurrentPPO
import matplotlib.pyplot as plt
import numpy as np
#from or_gym.envs.supply_chain.inventory_management import  LeanSupplyEnv0, LeanSupplyEnv01, AgileSupplyEnv0
from or_gym.envs.supply_chain.Batch_inventory_management import LeanSupplyEnv1, LeanSupplyEnv2, AgileSupplyEnv3
import time
import os
import argparse  
import json
import cloudpickle
from gymnasium.envs.registration import register,make



register(id='LeanSupplyEnv-v1',
	entry_point='or_gym.envs.supply_chain.Batch_inventory_management:LeanSupplyEnv1'
)



register(id='AgileSupplyEnv-v3',
	entry_point='or_gym.envs.supply_chain.Batch_inventory_management:AgileSupplyEnv3'
)




 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        models_dir = f"./tuning_models/{int(time.time())}/"
        logdir = f"./tuning_logs/{int(time.time())}/"

        create_directories(models_dir, logdir)

        for algorithm_name, algorithm_params_list in config.items():
            for params in algorithm_params_list:
                # log_name = params['log_name']
                environment_name = params['environment_name']
                learning_rate = params['learning_rate']
                # learning_rate = params.get('learning_rate', 0.0003) 
                # net_arch = params.get('net_arch')
                # n_lstm_layers = params.get('n_lstm_layers')
                policies_kwargs = get_policy_kwargs(algorithm_name,params )

                # 
                env = make(environment_name)
                env = DummyVecEnv([lambda: env])


                model = create_model(algorithm_name, env, logdir, policies_kwargs)
                log_name=get_tb_log_name(algorithm_name, params)


                
                
        

                TIMESTEPS = 1000
                total_iterations = 1000
                iters = 0
                for _ in range(total_iterations):
                    iters += 1
                    
                    print(f"Training {algorithm_name} on {environment_name}, log_name={log_name}")
                    model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False, tb_log_name=log_name)
                    model.save(f"{models_dir}/{TIMESTEPS*iters}")

    else:
        print("Please provide a configuration file using the --config argument.")

def get_tb_log_name(algorithm_name, params):
    learning_rate = params.get('learning_rate')
    net_arch = params.get('net_arch')
    n_lstm_layers = params.get('n_lstm_layers')
    
    if algorithm_name == "RecurrentPPO":
      
        return f'{algorithm_name}_{learning_rate}_{net_arch}_{n_lstm_layers}'
    else:
        return f'{algorithm_name}_{learning_rate}_{net_arch}_default'


def load_config(config_path):
    
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

def create_directories(models_dir, logdir):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
        
def get_policy_kwargs(algorithm_name, params):
    environment_name = params['environment_name']
    algorithm_name = params['algorithm_name']
    n_lstm_layers = params.get('n_lstm_layers', None)
    net_arch =params.get('net_arch', {})  
    learning_rate = params.get('learning_rate', 0.00003)

    print(f"algorithm_name: {algorithm_name}")
    print(f"learning_rate: {learning_rate}")
    print(f"net_arch: {net_arch}")
    print(f"n_lstm_layers: {n_lstm_layers}")


    combined_policy_kwargs = {
        'net_arch': {**net_arch, 'lstm': n_lstm_layers},
    }

    policies_kwargs = {
        'policy_kwargs': combined_policy_kwargs,
        'learning_rate': learning_rate,
        'verbose': 1,
    }
        

    return policies_kwargs

def create_model(algorithm_name, env, logdir, policies_kwargs):
    
    if algorithm_name == "PPO":
        print("Creating PPO model...")
        model =  PPO('MlpPolicy', env, tensorboard_log=logdir, **policies_kwargs)
        print(f"PPO model created with parameters: {policies_kwargs}")
        return model


    elif algorithm_name == "RecurrentPPO":
        print("Creating RecurrentPPO model...")
        model =  RecurrentPPO("MlpLstmPolicy", env, tensorboard_log=logdir,**policies_kwargs)
        print(f"RecurrentPPO model created with parameters: {policies_kwargs}")
        return model

    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


    
if __name__ == "__main__":
    main()    
        
        



        
                       
                        
