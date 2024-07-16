import numpy as np
import nengo
import math

from utils import sparsity_to_x_intercept

# from base import BaseModel

from tqdm import tqdm
import pandas as pd

from observation_wrappers import HexSSPObservation
from misc_wrappers import Monitor

class A2C(object):
    def __init__(self, 
                    env,
                    state_rep = 'HexSSP',
                    state_n_neurons = 3000,
                    #policy_n_params = 2,
                    active_prop = 0.1, 
                    neuron_type = nengo.RectifiedLinear(),
                    # alpha_actor = 0.02,
                    alpha_critic = 0.1,
                    gamma = 0.99,
                    show_progressbar = True
                    # beta = ?
                 ):
        
        # define the agent
        
        # to record the training data
        env = Monitor(env)
        self.env = env
        
        self.state_n_neurons = state_n_neurons
        self.active_prop = active_prop
        #self.policy_n_params = policy_n_params
        self.alpha_critic = alpha_critic
        self.gamma = gamma
        self.show_progressbar = show_progressbar
        
        # define the representation of the state
        
        if state_rep == 'HexSSP':
            self.env = HexSSPObservation(self.env)
        
        self.state_dimensions = len(self.env.observation_space.high)

        # shallow neural network for prediction and control
        with nengo.Network() as self.model:
            
            # subserve representation of the observation from environment
            self.ensemble = nengo.Ensemble( n_neurons = self.state_n_neurons, 
                                            dimensions = self.state_dimensions,
                                            neuron_type = neuron_type,
                                            intercepts = nengo.dists.Choice([sparsity_to_x_intercept(self.state_dimensions, self.active_prop)]),
                                          )

        self.sim = nengo.Simulator(self.model)
        
        # N x (# params)
        # self.policy_decoders = np.random.uniform(low = 0., high = 0.01, size = (self.state_n_neurons,policy_n_params) )
        self.value_decoders = np.zeros( (self.state_n_neurons,1) )
        
        # cache
        self.last_state = np.zeros( (self.state_dimensions) )
    
    # this should be the base interface for all non-hierarchical algorithms
    def learn( self, total_timesteps ):
        obs = self.env.reset()
        num_timesteps = 0
        
        if self.show_progressbar:
            progressbar = tqdm( total = total_timesteps )
            
        while num_timesteps < total_timesteps:
            action = np.random.normal()
        
            obs, reward, done, t, info = self.env.step(action)

            self.step(obs, reward, done)

            if done:
                obs = self.env.reset()
                
            else:
                num_timesteps += 1
                progressbar.update(1)
    
    def get_training_data(self):
        
        data = {
            'lengths'   : self.env.get_episode_lengths(),
            'rewards'   : self.env.get_episode_rewards()
        }
        
        epdf = pd.DataFrame( data = data )
        return epdf
        
    def get_training_metadata(self):
        # TODO -- print all nested attributes to a file?
        pass
    
    def get_activities(self,query_state):
        _,x = nengo.utils.ensemble.tuning_curves(self.ensemble, self.sim, query_state)
        return np.atleast_2d( x )
    
    def get_state_value(self,query_state):
        x = self.get_activities(query_state)
        return x @ self.value_decoders
        
    def compute_td_error(self, current_state, reward, done):
        last_state_value = self.get_state_value(query_state = self.last_state)

        td_error = reward - last_state_value
        if not done:
            current_state_value = self.get_state_value(query_state = current_state)
            td_error += self.gamma * current_state_value

        return td_error
        
    def update_value_decoders(self, td_error):
        x = self.get_activities(query_state = self.last_state)
        norm = np.clip( np.sum(x**2), a_min = 1e-16, a_max = None )
        dw = self.alpha_critic * td_error * x.T / norm
        self.value_decoders += dw
    
    def step(self, current_state, reward, done):
        td_error = self.compute_td_error(current_state, reward, done)
        last_state_value = self.get_state_value(query_state = self.last_state)

        self.update_value_decoders(td_error)
        self.last_state = current_state
    
    def get_policy_params(self,query_state):
        pass 
        # A = nengo.utils.ensemble.tuning_curves(self.sim.ensemble, query_state)
        # return A @ self.policy_decoders
    
    def update_policy_decoders(self, td_error):
        pass