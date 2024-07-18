import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import os

'''
Helper functions for the ChasingBearEnv environment
'''

def save_agent(agent, fname):
    '''
    save agent in a file. 
    '''
    filename = os.path.join('trained_models', fname)

    torch.save({
        'model_state_dict': agent.model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
    }, filename)


def load_agent(agent, filename):
    '''load the agent from a file.
    '''

    checkpoint = torch.load(filename)
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    agent.target_model.load_state_dict(checkpoint['model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.epsilon = checkpoint['epsilon']
    return agent


# Function to render an episode
def render_episode(env, agent, frames_dir, episode_num, render=True):
    '''
    Render an episode of the environment using the agent's policy
    '''
    state, _ = env.reset()
    state = state.flatten()
    total_reward = 0
    if render and not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    for time in range(env.max_steps):
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = next_state.flatten()
        state = next_state
        total_reward += reward
        if render:
            env.render(frame_num=time + episode_num * env.max_steps)
        if done:
            break
    return total_reward


# Create the animation using the saved PNG files
def create_animation(frames_dir, output_file, env):
    '''
    create animation
    '''
    fig = plt.figure()
    frames = [] 
    for i in range(env.max_steps):
        img = plt.imread(f'{frames_dir}/frame_{i:04d}.png')
        frame = plt.imshow(img, animated=True)
        frames.append([frame])
    ani = animation.ArtistAnimation(fig, frames, interval=200, repeat=False)
    ani.save(output_file, writer='pillow', fps=10)

# # Create animations
# create_animation('pre_training_frames', 'pre_training_animation.gif')
# create_animation('post_training_frames', 'post_training_animation.gif')


def remove_frames(frames_dir = 'frames'):
    '''
    Delete frames after the animation is rendered


























    
    '''

    # Delete all PNG files after creating the animation
    # Delete all files in the frames folder
    file_list = os.listdir(frames_dir)
    for file_name in file_list:
        file_path = os.path.join(frames_dir, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
