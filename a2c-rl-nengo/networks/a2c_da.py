import sys,os

sys.path.insert(0,os.path.join(os.path.dirname(__file__),'../representations'))
from observation_wrappers import HexSSPObservation

sys.path.insert(0,os.path.join(os.path.dirname(__file__),'../policies'))
from softmax_policy import SoftmaxPolicy

sys.path.insert(0,os.path.join(os.path.dirname(__file__),'../utils'))
from utils import ReLU, sparsity_to_x_intercept
from misc_wrappers import Monitor

from gymnasium import spaces
#import gym.spaces
from tqdm import tqdm
import pandas as pd
import numpy as np
import nengo
import math

class A2C(object):
    def __init__(self, 
                    env,
                    state_rep = 'HexSSP',
                    state_n_neurons = 4096,
                    active_prop = 0.242728, 
                    neuron_type = nengo.RectifiedLinear(),
                    alpha_actor = 0.195843,
                    alpha_critic = 0.195843,
                    gamma = 0.99,
                    show_progressbar = True
                    # beta = ?
                 ):
        
        # define the agent
        
        # to record the training data
        env = Monitor(env)
        self.env = env
        print(self.env.action_space)
        
        self.state_n_neurons = state_n_neurons
        self.active_prop = active_prop
        self.alpha_critic = alpha_critic
        self.alpha_actor = alpha_actor
        self.gamma = gamma
        self.show_progressbar = show_progressbar
        
        # define the representation of the state
        
        if state_rep == 'HexSSP':
            self.env = HexSSPObservation(self.env)
        
        self.state_dimensions = len(self.env.observation_space.high)

        # shallow neural network for prediction and control
        with nengo.Network() as self.model:
            
            # subserve representation of the observation from environment
            self.model.ensemble = nengo.Ensemble( n_neurons = self.state_n_neurons, 
                                            dimensions = self.state_dimensions,
                                            neuron_type = neuron_type,
                                            #gain = np.ones(self.state_n_neurons),
                                            #bias = np.zeros(self.state_n_neurons) - sparsity_to_x_intercept(self.state_dimensions, self.active_prop),
                                            #normalize_encoders = False,
                                            intercepts = nengo.dists.Choice([sparsity_to_x_intercept(self.state_dimensions, self.active_prop)]),
                                          )
                                          
        self.sim = nengo.Simulator(self.model)
        self.scaled_encoders = np.array( self.sim.data[self.model.ensemble].scaled_encoders.T )
        self.bias = np.array( self.sim.data[self.model.ensemble].bias.reshape(1,-1) )
        
        if isinstance(self.env.action_space,spaces.Discrete):# or isinstance(self.env.action_space,gym.spaces.Discrete):
            print('Creating softmax policy ... ')
            print('# actions: ', self.env.action_space.n)
            self.policy = SoftmaxPolicy( self.env.action_space.n )
        elif isinstance(self.env.action_space,spaces.Box):
            self.policy = GaussianPolicy( action_dim = len(self.env.observation_space.high) )
            
        mu = np.ones( (self.state_n_neurons,self.policy.n_params) ) * 0.1
        sigma = np.random.normal( loc = 0., scale = 0.02, size = (self.state_n_neurons,self.policy.n_params) )
        self.policy_param_decoders = mu + sigma
        #self.policy_param_decoders = np.random.normal( loc = 0, scale = 1., size = (self.state_n_neurons,self.policy.n_params) )
        self.value_decoders = np.zeros( (self.state_n_neurons,1) )
        
        # cache
        self.last_state = np.zeros( (self.state_dimensions) )
    
    # this should be the base interface for all non-hierarchical algorithms
    def learn( self, total_timesteps, seed = 0 ):
        #np.random.seed(seed)
        
        obs,_ = self.env.reset()
        num_timesteps = 0
        
        if self.show_progressbar:
            progressbar = tqdm( total = total_timesteps )
            
        while num_timesteps < total_timesteps:
            action = self.sample_action(obs)
        
            obs, reward, terminated, truncated, info = self.env.step(action)

            self.step(obs, reward, terminated, action)

            if terminated or truncated:
                obs,_ = self.env.reset()
                
            else:
                num_timesteps += 1
                progressbar.update(1)
    
    def sample_action(self,query_state):
        policy_params = self.get_policy_params(query_state)
        sample = self.policy.sample(policy_params)
        return sample
    
    def get_training_data(self):
        
        data = {
            'lengths'   : self.env.get_episode_lengths(),
            'rewards'   : self.env.get_episode_rewards()
        }
        
        epdf = pd.DataFrame( data = data )
        return epdf
        
    def get_training_data_continuing(self):
        
        data = {
            'reward'    : self.env.get_rewards()
        }
        
        print(len(data['reward']))
        
        epdf = pd.DataFrame( data = data )
        return epdf
        
    def get_training_metadata(self):
        # TODO -- print all nested attributes to a file?
        pass
        
    def save_model(self):
        # TODO -- dump scaled encoders, biases, and value/policy weights
        pass
    
    def get_activities(self,query_state):
        x = ReLU( self.scaled_encoders,self.bias,query_state )
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
    
    def step(self, current_state, reward, done, action):
        td_error = self.compute_td_error(current_state, reward, done)
        last_state_value = self.get_state_value(query_state = self.last_state)

        self.update_value_decoders(td_error)
        self.update_policy_decoders(td_error,action)
        self.last_state = current_state
    
    def get_policy_params(self,query_state):
        x = self.get_activities(query_state)
        return x @ self.policy_param_decoders

    def update_policy_decoders(self, td_error, action ):
        policy_params = self.get_policy_params(self.last_state)
        gradient_log_pi = self.policy.gradient_log_pi( action, policy_params )
        
        x = self.get_activities(query_state = self.last_state)
        norm = np.clip( np.sum(x**2), a_min = 1e-16, a_max = None )
        dw = self.alpha_actor * td_error * x.T / norm
        
        self.policy_param_decoders += dw @ gradient_log_pi