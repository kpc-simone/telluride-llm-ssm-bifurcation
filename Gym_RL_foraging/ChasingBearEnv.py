import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import matplotlib.animation as animation
import os

# Foraging environment class
class ChasingBearEnv(gym.Env):
    def __init__(self, max_steps=100):
        super(ChasingBearEnv, self).__init__()
        self.grid_size = 20  # Set the grid size to 20x20
        self.reward_block1 = (16, 16, 18, 18)  # Safehouse (2x2)
        self.reward_block2 = (0, 18, 2, 20)  # Safehouse (2x2)
        self.bear_start_area = (0, 0, 4, 4)  # Bear confined area (4x4 in top left)
        self.bear_pos = [1, 1]  # Initial position of the bear
        self.bear_size = 3  # Size of the bear sprite
        self.agent_size = 2  # Size of the agent sprite
        self.penalty_block_size = 6  # Size of the penalty block around the bear
        self.chase = False  # Flag to indicate if the bear is chasing the agent
        self.max_steps = 200  # Maximum number of steps per episode
        self.current_step = 0  # Current step count
        self.step_size = 2 # Step size of the agent

        
        self.state = np.zeros((self.grid_size, self.grid_size))
        self.agent_pos = [10, 10]
        self.trace = []  # List to store the trace of the agent's positions
        
        self.action_space = spaces.Discrete(4)  # 4 possible actions: up, down, left, right
        self.observation_space = spaces.Box(0, 1, shape=(self.grid_size, self.grid_size), dtype=np.float32)
        # self.serotonin_level = 1.0  # Start with high serotonin level
        
        self.bear_sprite = Image.open('sprites/black_bear.png')  # Load the bear sprite image
        self.safe_house_sprite = Image.open('sprites/BrickHouse.png')  # Load the safe house sprite image
        self.agent_sprite = Image.open('sprites/jl_sprite1.png')  # Load the agent sprite image

        self.is_caught = False  # Flag to indicate if the agent is caught by the bear
        
    def reset(self):
        self.agent_pos = [10, 10]
        self.bear_pos = [1, 1]
        self.state = np.zeros((self.grid_size, self.grid_size))
        self.trace = [self.agent_pos.copy()]  # Reset the trace list and add the initial position
        self.current_step = 0  # Reset the step count
        self.chase = False  # Reset the chase flag
        return self.state, {}
    
    def step(self, action):
        self.current_step += 1  # Increment the step count
        
        # Update the agent's position based on the action
        if action == 0:  # up
            self.agent_pos[0] = max(0, self.agent_pos[0] - self.step_size)
        elif action == 1:  # down
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + self.step_size)
        elif action == 2:  # left
            self.agent_pos[1] = max(0, self.agent_pos[1] - self.step_size)
        elif action == 3:  # right
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + self.step_size)
        
        # Add the new position to the trace
        self.trace.append(self.agent_pos.copy())

        # Check if the agent is in the safehouse 1
        if self.reward_block1[0] <= self.agent_pos[0] < self.reward_block1[2] and self.reward_block1[1] <= self.agent_pos[1] < self.reward_block1[3]:
            return self.state, 300, True, {}, {}  # Large reward and terminate
        
        # check if agent is in safehouse 2
        if self.reward_block2[0] <= self.agent_pos[0] < self.reward_block2[2] and self.reward_block2[1] <= self.agent_pos[1] < self.reward_block2[3]:
            return self.state, 300, True, {}, {}

        done = False  # Initialize the done flag
        reward = -1  # Set a small negative reward for each step to encourage faster completion
        
        # Check if the agent is within the penalty block around the bear
        penalty_block_x_start = max(0, self.bear_pos[0] - self.penalty_block_size // 2)
        penalty_block_x_end = min(self.grid_size - 1, self.bear_pos[0] + self.penalty_block_size // 2)
        penalty_block_y_start = max(0, self.bear_pos[1] - self.penalty_block_size // 2)
        penalty_block_y_end = min(self.grid_size - 1, self.bear_pos[1] + self.penalty_block_size // 2)
        
        if (penalty_block_x_start <= self.agent_pos[0] <= penalty_block_x_end and 
            penalty_block_y_start <= self.agent_pos[1] <= penalty_block_y_end):
            reward = -50  # Penalty for entering the penalty block
            self.chase = True  # Bear starts chasing the agent

        # Bear movement logic
        if self.chase:
            # Move the bear towards the agent: this part of the code can be tweaked
            if self.bear_pos[0] < self.agent_pos[0]:
                self.bear_pos[0] += 1
            elif self.bear_pos[0] > self.agent_pos[0]:
                self.bear_pos[0] -= 1
            if self.bear_pos[1] < self.agent_pos[1]:
                self.bear_pos[1] += 1
            elif self.bear_pos[1] > self.agent_pos[1]:
                self.bear_pos[1] -= 1
        else:
            # Random movement within the confined area
            bear_action = np.random.choice([0, 1, 2, 3])
            if bear_action == 0 and self.bear_pos[0] > 0:  # up
                self.bear_pos[0] -= 1
            elif bear_action == 1 and self.bear_pos[0] < self.bear_start_area[2] - 1:  # down
                self.bear_pos[0] += 1
            elif bear_action == 2 and self.bear_pos[1] > 0:  # left
                self.bear_pos[1] -= 1
            elif bear_action == 3 and self.bear_pos[1] < self.bear_start_area[3] - 1:  # right
                self.bear_pos[1] += 1
        
        # Check if the agent and bear occupy the same position
        if self.agent_pos == self.bear_pos:
            self.is_caught = True  # Set the flag to indicate that the agent is caught
            return self.state, -500, True, {}, {}  # Large penalty and terminate

        # Check if the maximum number of steps has been reached
        if self.current_step >= self.max_steps:
            done = True
        
        return self.state, reward, done, {}, {}  # Return the state, reward, done flag, and additional info
    
    def render(self, frame_num=None):
        fig, ax = plt.subplots()  # Create a new figure and axes
        ax.set_xlim(0, self.grid_size)  # Set the x-axis limits
        ax.set_ylim(0, self.grid_size)  # Set the y-axis limits

        # Draw the safehouse (reward block)
        # reward_patch = patches.Rectangle((self.reward_block1[1], self.reward_block1[0]), 
        #                                  self.reward_block1[3] - self.reward_block1[1], 
        #                                  self.reward_block[2] - self.reward_block1[0], 
        #                                  linewidth=1, edgecolor='g', facecolor='green')
        # ax.add_patch(reward_patch)  # Add the reward block to the plot

        # draw the safehouse (reward block)
        sh1 = ax.imshow(self.safe_house_sprite, extent=(self.reward_block1[1], self.reward_block1[3], self.reward_block1[0], self.reward_block1[2]), origin = 'lower')
        sh2 = ax.imshow(self.safe_house_sprite, extent=(self.reward_block2[1], self.reward_block2[3], self.reward_block2[0], self.reward_block2[2]), origin = 'lower')

        # Draw the bear sprite
        bear_img = ax.imshow(self.bear_sprite, extent=(self.bear_pos[1], self.bear_pos[1] + self.bear_size, self.bear_pos[0], self.bear_pos[0] + self.bear_size), origin = 'lower')

        # Draw the agent
        # ax.plot(self.agent_pos[1], self.agent_pos[0], 'bo')  # Plot the agent's position as a blue dot
        agent_img = ax.imshow(self.agent_sprite, extent=(self.agent_pos[1], self.agent_pos[1] + self.agent_size, self.agent_pos[0], self.agent_pos[0] + self.agent_size), origin = 'lower')

        # Draw the trace
        trace_x = [pos[1] for pos in self.trace]
        trace_y = [pos[0] for pos in self.trace]
        ax.plot(trace_x, trace_y, 'k', linewidth=1, alpha = 0.5)  # Plot the trace as a blue line

        plt.gca().invert_yaxis()  # Invert the y-axis to match grid coordinates
        plt.grid(True)  # Enable the grid 

        if frame_num is not None:
            if not os.path.exists('frames'):
                os.makedirs('frames')
            plt.savefig(f'frames/frame_{frame_num:04d}.png')

        # turn off the axes
        ax.axis('off')
        
        plt.close(fig)


# ## register the environment
# gym.envs.registration.register(
#     id='ChasingBear-v0',
#     entry_point='ChasingBearEnv:ChasingBearEnv',
#     max_episode_steps=200,
# )

# env = gym.make('ChasingBear-v0')