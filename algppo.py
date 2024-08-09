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
from supply_chain.Batch_inventory_management import LeanSupplyEnv1, LeanSupplyEnv2, AgileSupplyEnv3
import time
import os
import argparse
import json
import cloudpickle
from gymnasium.envs.registration import register,make



register(id='LeanSupplyEnv-v1',
	entry_point='supply_chain.Batch_inventory_management:LeanSupplyEnv1'
)

models_dir = f"tu_models/{int(time.time())}/"
logdir = f"tu_logs/{int(time.time())}/"


if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

env_name = 'LeanSupplyEnv-v1'
env = make(env_name)
# custom_policy_params = {
#     'n_lstm_layers': 2,  
# }
combined_policy_kwargs = {
    **dict(net_arch=dict(pi=[64,64], vf=[64,64]))
}

p_params = {
    'policy_kwargs':combined_policy_kwargs,
    'learning_rate': 0.003,
    'verbose': 1
}
model = PPO('MlpPolicy', env, tensorboard_log=logdir, **p_params)

TIMESTEPS = 1000
total_iterations = 1000
iters = 0
for _ in range(total_iterations):
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f'PPO_0.0003_pi=[64,64], vf=[64,64]')
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
    



