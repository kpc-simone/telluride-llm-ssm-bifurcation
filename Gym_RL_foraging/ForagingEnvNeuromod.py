import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys,os

#import seaborn as sns
from PIL import Image

import gymnasium as gym
from gymnasium import spaces

# set matplotlib font to arial
plt.rcParams['font.family'] = 'Arial'

class ForagingEnvNeuromod(gym.Env):
    def __init__(self, dopa_lvl=0.5, 
                    grid_size = 20, 
                    reward_block = (16, 16, 19, 19), 
                    penalty_block_size = 5, 
                    bear_pos = [5, 5], 
                    initial_agent_pos = [10, 10], 
                    step_size = 1, 
                    bear_size = 2
                    ):
        super(ForagingEnvNeuromod, self).__init__()

        self.grid_size = grid_size # set the grid size
        self.reward_block = reward_block # define a 4x4 reward block (x_start, y_start, x_end, y_end)
        self.state = np.zeros((2,)) # initialize the state
        self.agent_pos = initial_agent_pos # set the initial agent position
        self.action_space = spaces.Discrete(4) # define the action space: 4 actions (up, down, left, right)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32) # define the observation space
        
        self.dopa_level = dopa_lvl # set the initial dopamine level
        self.dopa_level_0 = dopa_lvl # set the initial dopamine level -- helpful in reset
        self.initial_agent_pos = initial_agent_pos.copy() # set the initial agent position -- helpful in reset
        self.trace = [] # traces to store the agent's path
        self.step_size = step_size # set the step size by which the agent moves
    

        ## bear params
        self.bear_size = bear_size # set the bear size
        self.bear_sprite = Image.open(os.path.join(os.path.dirname(__file__),'black_bear.png')) # load the bear sprite
        self.penalty_block_size = penalty_block_size # set the bear block size
        self.bear_pos = bear_pos # set the bear position

    def reset(self, seed = None, options = None):
        """
        Reset the environment and agent's position
        """
        self.state = np.asarray(self.initial_agent_pos.copy())
        self.dopa_level = self.dopa_level_0
        self.trace.append(self.agent_pos.copy())
        return self.state, {} # return the state
    
    def step(self, action):
        '''
        Take an action and update the environment.
        1. [0] controls up/down
        2. [1] controls left/right
        '''
        # update agent's position based on the action
        # up
        if action == 0:
            self.state[0] = max(0, self.state[0] - self.step_size)
        # down
        elif action == 1:
            self.state[0] = min(self.grid_size - 1, self.state[0] + self.step_size)

        # left
        elif action == 2:
            self.state[1] = max(0, self.state[1] - self.step_size)

        # right
        elif action == 3:
            self.state[1] = min(self.grid_size - 1, self.state[1] + self.step_size)

        # store the trace
        self.trace.append(self.state.copy())

        # TODO: Bear movement

        ## set done flag
        done = False
        reward = -1 # default reward

        # draw a penalty block around the bear
        penalty_block_x_start = self.bear_pos[0] - (self.penalty_block_size // 2)
        penalty_block_x_end = self.bear_pos[0] + (self.penalty_block_size // 2)
        penalty_block_y_start = self.bear_pos[1] - (self.penalty_block_size // 2)
        penalty_block_y_end = self.bear_pos[1] + (self.penalty_block_size // 2)

        ## check if the agent is in the reward block
        if self.reward_block[0] <= self.state[0] <= self.reward_block[2] and self.reward_block[1] <= self.state[1] <= self.reward_block[3]:
            reward = 100
            done = True

        ## check if the agent is in the penalty block
        elif penalty_block_x_start <= self.state[0] <= penalty_block_x_end and penalty_block_y_start <= self.state[1] <= penalty_block_y_end:
            reward = -100  # Set a large negative reward
            done = True  # Set the done flag to True

        ## TODO: update dopa level if needed
        print(self.state,reward)
        return self.state, reward, done, {}, {}
    
    def render(self, mode = 'human'):
        '''
        Render the environment
        '''

        fig, ax = plt.subplots(1,1)
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)

        # draw the reward block
        reward_patch = mpatches.Rectangle((self.reward_block[0], self.reward_block[1]), self.reward_block[2] - self.reward_block[0], self.reward_block[3] - self.reward_block[1], color='green')
        ax.add_patch(reward_patch)

        # draw the penalty block
        penalty_patch = mpatches.Rectangle((self.bear_pos[1] - 2, self.bear_pos[0] - 2), self.penalty_block_size, self.penalty_block_size, linewidth=1, edgecolor='r', facecolor='none', linestyle='--')
        ax.add_patch(penalty_patch)  # Add the penalty block to the plot

        # draw the bear
        im = ax.imshow(self.bear_sprite, extent=(self.bear_pos[1], self.bear_pos[1] + self.bear_size, self.bear_pos[0], self.bear_pos[0] + self.bear_size), origin = 'lower')

        # Draw the penalty block around the bear sprite
        penalty_patch = mpatches.Rectangle((self.bear_pos[1] - 2, self.bear_pos[0] - 2), self.penalty_block_size, self.penalty_block_size, linewidth=1, edgecolor='r', facecolor='none', linestyle='--')
        ax.add_patch(penalty_patch)  # Add the penalty block to the plot

        # draw agent's initial position
        ax.plot(self.initial_agent_pos[1], self.initial_agent_pos[0], 'rx', markersize = 10)

        # draw the agent
        ax.plot(self.agent_pos[1], self.agent_pos[0], 'ko', label = 'Agent')

        # plot the traces
        trace_x = [x[1] for x in self.trace]
        trace_y = [x[0] for x in self.trace]
        ax.plot(trace_x, trace_y, c = 'k', label = 'Traces', alpha = 0.3)

        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.gca().invert_yaxis()
        # plt.axis('off')
        plt.grid(True)
        # plt.legend()


        # sns.despine()


        plt.show()

## test the environment: Remove later

if __name__ == "__main__":

    env =  ForagingEnvNeuromod()
    env.reset()
    # env.render()

    # # simulate 1 step
    # env.step(1)  # Move the agent down
    # env.render()  # Render the environment

    # env.step(3)  # Move the agent down
    # env.render()  # Render the environment

    steps_arr = np.random.randint(0, 4, 1000)
    for step in steps_arr:
        env.step(step)

    env.render()  # Render the environment
