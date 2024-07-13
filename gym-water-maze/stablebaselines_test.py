from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import os
import gymnasium as gym
import gym_water_maze

# env_name = 'WaterMaze-v0'
# n_train_steps = 10000

env_name = 'ForageWaterMaze-v0'
n_train_steps = 50000

# env_name = 'RelativeWaterMaze-v0'
# n_train_steps = 50000

reward_type = "active"
env = gym.make(env_name,render_trace=True, reward_type=reward_type)
obs, _ = env.reset()
for i in range(100):
    a = 2*np.random.rand(2) - 1
    _,_,_,_,_ = env.step(a)
env.render()

env = gym.make(env_name,render_trace=True, reward_type=reward_type)
log_dir = "./watermazetorch/"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)
set_random_seed(0)
model = PPO(MlpPolicy, env, verbose=0, learning_rate=0.0003,
            n_steps=512, batch_size=8, n_epochs=20, gamma=0.99,
            gae_lambda=0.95, clip_range=0.2, clip_range_vf=None,
            normalize_advantage=True, ent_coef=3.0e-5, vf_coef=0.2,
            max_grad_norm=0.8)

# Use a separate environement for evaluation
eval_env = gym.make(env_name, render_mode="rgb_array", reward_type=reward_type)

# Random Agent, before training
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

print(f"mean reward before training:{mean_reward:.2f} +/- {std_reward:.2f}")

# Train the agent for 10000 steps
model.learn(total_timesteps=n_train_steps)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)

print(f"mean reward after training:{mean_reward:.2f} +/- {std_reward:.2f}")

def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')

    # y_ma = moving_average(y, window=50)
    fig = plt.figure(title)
    plt.plot(x, y, color='grey' )
    # plt.plot(x[len(x) - len(y_ma):], y_ma, color='blue' )
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()

plot_results(log_dir, title='Stable baseline PPO on Water Maze')
