from stable_baselines3.common.evaluation import evaluate_policy
import os
import time
import argparse  #
import json

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



# Define the list of environment names
batch_demand_env_names = [
   "LeanSupplyEnv-v1",
    "LeanSupplyEnv-v2",
    "AgileSupplyEnv-v3"
]


reward_records = {'PPO': {f'env{i}': {'steps': [], 'reward': []} for i, _ in enumerate(batch_demand_env_names)}}




total_steps_per_env = 10**5
log_interval = 1000
total_learning_cycles = 3  
start_time = time.time()

for cycle in range(total_learning_cycles):
    for env_idx, env_name in enumerate(batch_demand_env_names):
        
        
         # Initialize the environment based on environment_name     
        if env_name  == "AgileSupplyEnv-v3":
            env = make('AgileSupplyEnv-v3')
        elif env_name  == "LeanSupplyEnv-v2":
            env = make('LeanSupplyEnv-v2')
        elif env_name  == "LeanSupplyEnv-v1":
            env = make('LeanSupplyEnv-v1')
        

        
        else:
            print(f"Unknown environment:{env_name}")
            continue

        
        
        model = PPO('MlpPolicy', env, verbose=1)
        steps = 0
        reward_history = {'steps': [], 'reward': []}
        
        while steps < total_steps_per_env:
            model.learn(total_timesteps=log_interval, reset_num_timesteps=False)
            steps += log_interval

            elapsed_time = time.time() - start_time
            mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
            reward_records[env_name][f'env{env_idx}']['steps'].append(steps)
            reward_records[env_name][f'env{env_idx}']['reward'].append(mean_reward)
            print(f"Mean reward for {env_name} (PPO) after {steps} steps: {mean_reward}")
            
            
        if env_name not in reward_records:
            reward_records[env_name] = {'steps': [], 'reward': []}

        reward_records[env_name]['steps'].append(reward_history['steps'])
        reward_records[env_name]['reward'].append(reward_history['reward'])
        # Save the reward history for this environment and cycle
        filename = f'reward_{env_name}_cycle_{cycle + 1}.npy'
        np.save(filename, reward_history)
        
        
        
        env.close()

# Save the overall reward records
with open('reward_records.json', 'w') as f:
    json.dump(reward_records, f)
    
    
# Create individual reward plots for each environment
colors = ['b', 'r', 'y']
for env_name in stochastic_demand_env_names:
    
    plt.figure(figsize=(12, 6))
    
    env_indices = [i for i, name in enumerate(reward_records['PPO']['env_name']) if name == env_name]

    steps =  [reward_records['PPO']['steps'][i] for i in env_indices]
    rewards = [reward_records['PPO']['reward'][i] for i in env_indices]
    color = colors[stochastic_demand_env_names.index(env_name) % len(colors)]
    plt.plot(steps, rewards, label=f'{env_name}', alpha=0.5,color=color)





    plt.xlabel('Steps')
    plt.ylabel('Mean Reward')
    plt.legend(loc='upper left')
    plt.title(f'Mean Reward History for {env_name}')
    plt.savefig(f'./logss/reward_{env_name}_cycle_{cycle + 1}.png')


plt.close('all')


combined_reward_data = []
# Load the data from the nine separate reward history files
reward_history_files = [
    'reward_LeanSupplyEnv-v3_cycle_1.npy',
    'reward_LeanSupplyEn-v2_cycle_1.npy',
    'reward_AgileSupplyEnv-v1_cycle_1.npy',
    'reward_LeanSupplyEnv-v3_cycle_2.npy',
    'reward_LeanSupplyEn-v2_cycle_2.npy',
    'reward_AgileSupplyEnv-v1_cycle_2.npy',
    'reward_LeanSupplyEnv-v3_cycle_3.npy',
    'reward_LeanSupplyEn-v2_cycle_3.npy',
    'reward_AgileSupplyEnv-v1_cycle_3.npy'
]


# Create a list to store the reward data
reward_data = []

# Load the data from the reward history files
for file in reward_history_files:
    data = np.load(file, allow_pickle=True).item()
    reward_data.append(data['PPO']['reward'])
    
    
# Transpose the data to have it in the right format for plotting
reward_data = np.array(reward_data).T

# Plot the combined reward data
plt.figure(figsize=(12, 6))
colors = ['b', 'r', 'y']
labels = ['LeanSupplyEnv-v0', 'LeanSupplyEn-v0', 'AgileSupplyEnv-v0']

for i in range(3):
    plt.plot(reward_data[:, i], label=labels[i], color=colors[i])

  
    
    
# Customize the plot
plt.xlabel('Steps')
plt.ylabel('Mean Reward')
plt.legend(loc='upper left')
plt.title('Combined Mean Reward History for Stochastic Demand Environments')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))



# Save the final combined plot
plt.savefig('./logss/combined_reward_batch.png')




