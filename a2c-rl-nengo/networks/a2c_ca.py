import numpy as np
import nengo
import math

from utils import sparsity_to_x_intercept

# from base import BaseModel

from gymnasium import spaces
from tqdm import tqdm
import pandas as pd

from policies import SoftmaxPolicy, GaussianPolicy
from utils import ReLU

from observation_wrappers import HexSSPObservation
from misc_wrappers import Monitor

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
                                            intercepts = nengo.dists.Choice([sparsity_to_x_intercept(self.state_dimensions, self.active_prop)]),
                                          )

        self.sim = nengo.Simulator(self.model)
        self.scaled_encoders = np.array( self.sim.data[self.model.ensemble].scaled_encoders.T )
        self.bias = np.array( self.sim.data[self.model.ensemble].bias.reshape(1,-1) )
        
        if isinstance(self.env.action_space,spaces.Discrete):
            print('Creating softmax policy ... ')
            self.policy = SoftmaxPolicy( self.env.action_space.n )
            self.policy_param_decoders = np.zeros( (self.state_n_neurons,self.policy.params_per_dim) )
        elif isinstance(self.env.action_space,spaces.Box):
            self.policy = GaussianPolicy( action_dim = len(self.env.action_space.low) )
            self.policy_param_decoders = np.zeros( (self.state_n_neurons, self.policy.action_dim, self.policy.params_per_dim ) )
            # print('decoders shape at policy creation: ', self.policy_param_decoders.shape)

        self.value_decoders = np.zeros( (self.state_n_neurons,1) )
        
        # cache
        self.last_state = np.zeros( (self.state_dimensions) )
    
    # this should be the base interface for all non-hierarchical algorithms
    def learn( self, total_timesteps ):
        obs,_ = self.env.reset()
        num_timesteps = 0
        
        if self.show_progressbar:
            progressbar = tqdm( total = total_timesteps )
            
        while num_timesteps < total_timesteps:
            action = self.sample_action(obs)
            # print('action shape: ', action.shape)
        
            obs, reward, terminated, truncated, info = self.env.step(action)

            self.step(obs, reward, terminated, action)

            if terminated or truncated:
                obs,_ = self.env.reset()
                
            else:
                num_timesteps += 1
                progressbar.update(1)
    
    def sample_action(self,query_state):
        policy_params = self.get_policy_params(query_state)
        #print('params, sampling: ', policy_params)
        
        sample = self.policy.sample(policy_params)
        return sample
    
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
        #print(x.shape,self.policy_param_decoders.shape)
        
        if len(self.policy_param_decoders.shape) == 2:
            return x @ self.policy_param_decoders
        else:
            #policy_params = np.zeros( (self.policy.action_dim,self.policy.params_per_dim) )
            #for d in range(self.policy.action_dim):
            #    policy_params[:,d] = x @ self.policy_param_decoders[:,:,d]
            
            return np.tensordot( x, self.policy_param_decoders, axes = 1 )[0,:,:]

    def update_policy_decoders(self, td_error, action ):
        policy_params = self.get_policy_params(self.last_state)
        # print('pp shape: ', policy_params.shape )
        
        gradient_log_pi = self.policy.gradient_log_pi( action, policy_params )
        # print('glp shape: ', gradient_log_pi.shape)
        
        x = self.get_activities(query_state = self.last_state)
        norm = np.clip( np.sum(x**2), a_min = 1e-16, a_max = None )
        dw = self.alpha_actor * td_error * x.T / norm
        # print('dw shape: ', dw.shape)
        
        if len(self.policy_param_decoders.shape) == 2:
            self.policy_param_decoders += dw @ gradient_log_pi
        else:
            for p in range(self.policy.params_per_dim):
                #print('gradient_log_pi[:,p] shape: ', gradient_log_pi[:,p].reshape(1,-1).shape)
                self.policy_param_decoders[:,:,p] += dw @ gradient_log_pi[:,p].reshape(1,-1)