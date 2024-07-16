
import gymnasium as gym
from gymnasium import spaces

import numpy as np
import math
import gym

class Walk1D(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,
            min_state_ = 0.,
            max_state_ = 100.,
            actions = [-1,1],
            states_reward_ = [40,60],
            state_start_ = 5,
            max_steps_ = 100
            ):
            
        # above may be cause of toggling between adjacent states
        self.observation_space = spaces.Box( low = min_state_, high = max_state_ , shape = (1,) )
        self.action_space = spaces.Discrete(2)#spaces.Box( low = actions[0], high = actions[1], shape = (1,) )
        
        self.min_state = min_state_
        self.max_state = max_state_
        self.reward_states = states_reward_
        
        self.state_start = np.random.randint( low = self.min_state, high = self.max_state-1)
        self.state = self.state_start
        self.next_state = self.state_start
        self.reward_counter = 0
        self.step_count = 0
        self.max_steps = max_steps_
            
    def reset(self,seed = None, options = {}):
        
        # print('initializing for new episode')
        # in_reward_state = True
        # while in_reward_state:
            # for reward_state in self.reward_states:
                # #print(self.state_start,reward_state)
                # if np.abs(self.state_start - reward_state) < 1.5:
                    # #print('reward state identified')
                    # in_reward_state = True
                    # break
                # else:
                    # in_reward_state = False
        
        self.state_start = np.random.uniform( low = self.min_state, high = self.max_state-1 )
        self.state = self.state_start
        self.next_state = self.state_start
        
        temp = self.reward_states.copy()
        ds1 = [ np.abs(reward_state - self.state) for reward_state in temp ]
        temp = [ 100-r for r in temp ]
        ds2 = [np.abs(reward_state+self.state) for reward_state in temp]
        
        distances = ds1 + ds2        
        self.least_steps = math.ceil( min( distances ) ) 
        
        self.done = False
        self.reward = 0.
        self.reward_counter = 0
        self.step_count = 0
        
        return np.array(self.state),{}
    
    def step(self,action):
        self.step_count += 1
        
        if action == 1:
            action_ = -1
        else:
            action_ = 1
        
        self.next_state = self.state + action_
        for reward_state in self.reward_states:
            if np.abs(self.next_state - reward_state) < 1.5:      
                
                # TODO: modify discounting to compute from a sigmoid; ie steps < least steps return a maximum reward
                
                if self.step_count < self.least_steps:
                    self.reward = 1.0
                else:
                    self.reward = 1.0 * self.least_steps / self.step_count
                self.reward_counter += 1
                
            else:
                self.reward = 0.
        
        if self.reward_counter >= 1:
            self.done = True
        elif self.step_count > self.max_steps:
            self.done = True            
            
        if self.next_state < 0:
            self.next_state = self.max_state - action
        elif self.next_state > self.max_state:
            self.next_state = action
            
        self.state = self.next_state
        
        return np.array(self.next_state),self.reward,self.done,False,{}