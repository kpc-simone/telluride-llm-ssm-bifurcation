import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import gym

import gym
from gym import spaces

# set matplotlib font to arial
plt.rcParams['font.family'] = 'Arial'

class ForagingEnv(gym.Env):
    def __init__(self, ser_lvl=0.5, grid_size=200, reward_block=(180, 180, 184, 184), penalty_block=(40, 40, 54, 54), initial_agent_pos = [100, 100], step_size = 5):
        super(ForagingEnv, self).__init__()

        self.grid_size = grid_size # set the grid size
        self.reward_block = reward_block # define a 4x4 reward block (x_start, y_start, x_end, y_end)
        self.penalty_block = penalty_block # define as 4x4 penalty block (x_start, y_start, x_end, y_end)
        self.state = np.zeros((self.grid_size, self.grid_size)) # initialize the state
        self.agent_pos = initial_agent_pos # set the initial agent position
        self.action_space = spaces.Discrete(4) # define the action space: 4 actions (up, down, left, right)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.float32) # define the observation space
        self.serotonin_level = ser_lvl # set the initial serotonin level
        self.serotonin_level_0 = ser_lvl # set the initial serotonin level -- helpful in reset
        self.initial_agent_pos = initial_agent_pos.copy() # set the initial agent position -- helpful in reset
        self.trace = [] # traces to store the agent's path
        self.step_size = step_size # set the step size by which the agent moves

    def reset(self):
        """
        Reset the environment and agent's position
        """
        self.state = np.zeros((self.grid_size, self.grid_size))
        self.agent_pos = self.initial_agent_pos.copy()
        self.serotonin_level = self.serotonin_level_0
        self.trace.append(self.agent_pos.copy())
        return self.state # return the state
    
    def step(self, action):
        '''
        Take an action and update the environment.
        1. [0] controls up/down
        2. [1] controls left/right
        '''
        # update agent's position based on the action
        # up
        if action == 0:
            self.agent_pos[0] = max(0, self.agent_pos[0] - self.step_size)
        # down
        elif action == 1:
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + self.step_size)

        # left
        elif action == 2:
            self.agent_pos[1] = max(0, self.agent_pos[1] - self.step_size)

        # right
        elif action == 3:
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + self.step_size)

        # store the trace
        self.trace.append(self.agent_pos.copy())

        ## set done flag
        done = False
        reward = -1 # default reward

        ## check if the agent is in the reward block
        if self.reward_block[0] <= self.agent_pos[0] <= self.reward_block[2] and self.reward_block[1] <= self.agent_pos[1] <= self.reward_block[3]:
            reward = 100
            done = True

        ## check if the agent is in the penalty block
        if self.penalty_block[0] <= self.agent_pos[0] <= self.penalty_block[2] and self.penalty_block[1] <= self.agent_pos[1] <= self.penalty_block[3]:
            reward = -100
            done = True


        ## TODO: update serotonin level if needed

        return self.state, reward, done, {}
    
    def render(self, mode = 'human'):
        '''
        Render the environment
        '''

        fig, ax = plt.subplots()
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)

        # draw the reward block
        reward_patch = mpatches.Rectangle((self.reward_block[0], self.reward_block[1]), self.reward_block[2] - self.reward_block[0], self.reward_block[3] - self.reward_block[1], color='green')
        ax.add_patch(reward_patch)

        # draw the penalty block
        penalty_patch = mpatches.Rectangle((self.penalty_block[0], self.penalty_block[1]), self.penalty_block[2] - self.penalty_block[0], self.penalty_block[3] - self.penalty_block[1], color='red')
        ax.add_patch(penalty_patch)

        # draw agent's initial position
        ax.plot(self.initial_agent_pos[1], self.initial_agent_pos[0], 'rx', markersize = 20)

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

env =  ForagingEnv()
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
