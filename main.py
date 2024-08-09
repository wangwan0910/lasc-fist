# -*- coding: utf-8 -*-


import os
import argparse  
import json
import or_gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
import cloudpickle
from sb3_contrib import RecurrentPPO
import matplotlib.pyplot as plt
import numpy as np
from or_gym.envs.supply_chain.inventory_management import  LeanSupplyEnv0, LeanSupplyEnv01, AgileSupplyEnv0
from or_gym.envs.supply_chain.Batch_inventory_management import LeanSupplyEnv1, LeanSupplyEnv2, AgileSupplyEnv3
from gymnasium.envs.registration import register,make
import time

register(id='LeanSupplyEnv-v0',
	entry_point='or_gym.envs.supply_chain.inventory_management:LeanSupplyEnv0'
)


register(    
    id="LeanSupplyEn-v0", 
    entry_point="or_gym.envs.supply_chain.inventory_management:LeanSupplyEnv01"
)


register(
   
    id="AgileSupplyEnv-v0",
    
    entry_point="or_gym.envs.supply_chain.inventory_management:AgileSupplyEnv0"
    
)

register(id='LeanSupplyEnv-v1',
	entry_point='or_gym.envs.supply_chain.Batch_inventory_management:LeanSupplyEnv1'
)


register(id='LeanSupplyEnv-v2',
 	entry_point='or_gym.envs.supply_chain.Batch_inventory_management:LeanSupplyEnv2'
 )
 
register(id='AgileSupplyEnv-v3',
	entry_point='or_gym.envs.supply_chain.Batch_inventory_management:AgileSupplyEnv3'
)




def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

def create_directories(models_dir, logdir):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        models_dir = f"./eval_models/{int(time.time())}/"
        logdir = f"./eval_logs/{int(time.time())}/"

        create_directories(models_dir, logdir)

        for demand_type, demand_config in config.items():
            for algorithm_config in demand_config:
                algorithm_name = algorithm_config['algorithm']
                for environment_config in algorithm_config['environments']:
                    environment_name = environment_config['name']
                    log_name = environment_config['log_name']

                    # Initialize the environment based on environment_name
                    if environment_name == "LeanSupplyEnv-v0":
                        env = make('LeanSupplyEnv-v0')
                    elif environment_name == "LeanSupplyEn-v0":
                        env = make('LeanSupplyEnv-v0')
                    elif environment_name == "AgileSupplyEnv-v0":
                        env = make('AgileSupplyEnv-v0')
                    elif environment_name == "LeanSupplyEnv-v1":
                        env = make('LeanSupplyEnv-v1')
                    elif environment_name == "LeanSupplyEnv-v2":
                        env = make('LeanSupplyEnv-v2')
                    elif environment_name == "AgileSupplyEnv-v3":
                        env = make('AgileSupplyEnv-v3')
                    else:
                        print(f"Unknown environment: {environment_name}")
                        continue

                    if algorithm_name == "PPO":
                        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
                    elif algorithm_name == "RecurrentPPO":
                        model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=logdir)
                    else:
                        print(f"Unknown algorithm: {algorithm_name}")
                        continue

                    TIMESTEPS = 30
                    total_iterations = 30
                    iters = 0
                    for _ in range(total_iterations):
                        iters += 1
                        print(f"Training {algorithm_name} on {environment_name}, log_name={log_name}")
                        model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False, tb_log_name=log_name)
                        model.save(f"{models_dir}/{TIMESTEPS*iters}")

    else:
        print("Please provide a configuration file using the --config argument.")

if __name__ == "__main__":
    main()

