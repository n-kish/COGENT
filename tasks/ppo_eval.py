import os
import argparse
import json

import gymnasium as gym
from sbx import PPO
# from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv  # Add this import
import multiprocessing
from stable_baselines3.common.monitor import Monitor
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean



import fcntl
import time
import re
import math
import numpy as np
from matplotlib import pyplot as plt

def main():
    # print("Entered runppo")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--env_id', help='environment ID', default="Hopper-v5")
    parser.add_argument('--num_timesteps', help='number of timesteps to train the robot for', type=int, default=10_000_000)
    parser.add_argument('--expt_name', help='name of the experiment', type=str, default='test')
    # parser.add_argument('--xml_file_path', help='path for xml', default="./assets/base_hopper_flat.xml")
    parser.add_argument('--xml_file_path', help='path for xml', default='/scratch/knagiredla/robonet/logs/exp_GSCA_5_flat_base_hopper_50_000_134_1734445021/train/robot_1_62500.0_1734458559.xml')
    parser.add_argument('--results_path', help='path for xml', default="./results")
    parser.add_argument('--ctrl_cost_weight', help='ctrl cost weight for gym env', type=float, default=0.0005)
    parser.add_argument('--n_envs', help='number of environments', type=int, default=8)

    args = parser.parse_args()


    env_id = args.env_id

    # model_path = os.path.join(args.results_path, f"{args.expt_name}-{args.xml_file_path[-25:]}.json")
    # if not os.path.exists(model_path):
    #     os.makedirs(model_path)
    # new_logger = configure(model_path, ["stdout", "csv", "tensorboard"])

    policy_kwargs = dict(
            net_arch=[128, 128, 128, 128]
        )

    expt_path = f'./{args.results_path}/{args.expt_name}/'

    n_envs = args.n_envs
    vec_env = SubprocVecEnv([
        lambda i=i: Monitor(
            gym.make(env_id,
                xml_file=args.xml_file_path,
                render_mode='rgb_array',
                width=1920,
                height=1080,
                camera_id=0),
            os.path.join(expt_path, str(i))
        ) for i in range(n_envs)
    ])

    plots_path = f'./{args.results_path}/{args.expt_name}/plots/'
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    if env_id == "Swimmer-v5":
        model = PPO("MlpPolicy", vec_env, verbose=1, batch_size=2048, learning_rate=0.0001, 
                    clip_range=0.1, ent_coef=0.01, policy_kwargs=policy_kwargs, gamma=1, tensorboard_log=plots_path)
    else:
        model = PPO("MlpPolicy", vec_env, verbose=1, batch_size=2048, learning_rate=0.0001, 
                    clip_range=0.1, ent_coef=0.01, policy_kwargs=policy_kwargs, tensorboard_log=plots_path)   
    
    model.learn(total_timesteps=args.num_timesteps, progress_bar=True, tb_log_name=f'{args.xml_file_path[-25:]}')  
    # mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
    
    model_dir = os.path.join(expt_path, "models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  # Ensure the model directory exists
    model.save(os.path.join(model_dir, f"{args.expt_name}-{args.xml_file_path[-25:]}.zip"))  # Save the model

if __name__ == '__main__':
    main()
