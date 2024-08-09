
from stable_baselines3.common.evaluation import evaluate_policy

import time
import argparse  #
import json
import os
import random as rd
import gymnasium as gym
import numpy as np
import or_gym
from or_gym.utils import create_env
from or_gym.envs.env_list import ENV_LIST
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
import cloudpickle
from sb3_contrib import RecurrentPPO
import matplotlib.pyplot as plt
import numpy as np
from or_gym.envs.supply_chain.inventory_management import  LeanSupplyEnv0, LeanSupplyEnv01, AgileSupplyEnv0
from or_gym.envs.supply_chain.Batch_inventory_management import LeanSupplyEnv1, LeanSupplyEnv2, AgileSupplyEnv3

from matplotlib.ticker import MaxNLocator
from gymnasium.envs.registration import register,make
from stable_baselines3.common.monitor import Monitor
from matplotlib.ticker import MaxNLocator
from gymnasium.envs.registration import register,make
register(id='LeanSupplyEnv-v0',
	entry_point='or_gym.envs.supply_chain.inventory_management:LeanSupplyEnv0'
)


register(    
    id="LeanSupplyEn-v0", 
    entry_point="or_gym.envs.supply_chain.inventory_management:LeanSupplyEnv01"
)

register(
    # unique identifier for the env `name-version`
    id="AgileSupplyEnv-v0",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point="or_gym.envs.supply_chain.inventory_management:AgileSupplyEnv0"
    # Max number of steps per episode, using a `TimeLimitWrapper`
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



stochastic_demand_env_names = [
    "AgileSupplyEnv-v0",
    "LeanSupplyEn-v0",
    "LeanSupplyEnv-v0",
    "AgileSupplyEnv-v0",
    "LeanSupplyEn-v0",
    "LeanSupplyEnv-v0",
    "AgileSupplyEnv-v0",
    "LeanSupplyEn-v0",
    "LeanSupplyEnv-v0"
]
# Define the list of environment names
# batch_demand_env_names = [
#     "AgileSupplyEnv-v3",
#     "LeanSupplyEnv-v2",
#     "LeanSupplyEnv-v1",
#     "AgileSupplyEnv-v3",
#     "LeanSupplyEnv-v2",
#     "LeanSupplyEnv-v1",
#      "AgileSupplyEnv-v3",
#     "LeanSupplyEnv-v2",
#     "LeanSupplyEnv-v1"
# ]
# Extract unique environment names
# unique_env_names = list(set(batch_demand_env_names))
# print(unique_env_names)


#reward_records = {'PPO': {f'env{i}': {'steps': [], 'reward': []} for i, _ in enumerate(stochastic_demand_env_names)}}

#reward_records = {'PPO': {'steps': [], 'reward': [], 'env_name': []}}
reward_records = {env_name: {'steps': [], 'reward': []} for env_name in stochastic_demand_env_names}


total_steps_per_env = 10**5
log_interval = 1000
total_learning_cycles = 3  
start_time = time.time()



def create_env(env_name):
    env = make(env_name)
    return env



log_dir = "loggs/reppo/"
os.makedirs(log_dir, exist_ok=True)
# Create an agent
# model = PPO("MlpPolicy", create_env(stochastic_demand_env_names[0]), verbose=1)
model = RecurrentPPO('MlpLstmPolicy', create_env(stochastic_demand_env_names[0]), verbose=1)
# env = create_env()
# env = Monitor(env, log_dir+"_test_0")
# obs = env.reset()
# model = PPO("MlpPolicy", env,
#             verbose=1
#             )
#CartPole has 200 max steps
model.learn(total_timesteps=20000, log_interval=1000)
model.save('./loggs/reppo/ppo_save')

# del env
del model
data = []
# Load and Train
for i, env_name in enumerate(stochastic_demand_env_names):
    env = create_env(env_name)
    env = Monitor(env, os.path.join(log_dir, f"test_{i}"))
    
    model = PPO.load("loggs/reppo/ppo_save")
   # model.load_replay_buffer('logs/dqn_save_replay_buffer')
    
    model.set_env(env)
    # model.set_random_seed(seeds[i])
    
    model.learn(total_timesteps=20000, log_interval=1000,reset_num_timesteps=False)
    episode_rewards = env.get_episode_rewards()
    data.append(episode_rewards)
    model.save('loggs/reppo/ppo_save')
    del model
    del env


