# hexssp_obs_wrapper.py
import numpy as np
import sys,os

import gymnasium
from gymnasium import spaces
import sspspace

# other wrappers?
# Fourier basis functions
# OneHot representation
# Random SSP

class HexSSPObservation( gymnasium.ObservationWrapper ):
    def __init__(self, env: gymnasium.Env, length_scale = 'auto', n_rotates = 8, n_scales = 5, scale_min = 0.1, scale_max = 10):
        
        '''
        Transforms observation space of the gym-maze environment to the HexSSP representation

        Args:
            env: The environment to apply the wrapper
            length_scale: the length scale of the SSP representation
            n_rotates: the number of rotations to apply to V when constructing the HexSSP phase matrix
            n_scales: the number of scales to apply to V when constructing the HexSSP phase matrix
            scale_min, scale_max: the minimum and maximum scales to apply to V when constructing the HexSSP phase matrix
        '''
        
        super().__init__(env)
        low = self.observation_space.low
        high = self.observation_space.high
        domain_dim = len(low)
            
        if length_scale == 'auto':
            length_scale = np.clip( ( np.atleast_2d( high - low ) ).T,a_min = 0., a_max = 10) / 10.
          
        print(length_scale)
        self.rep = sspspace.HexagonalSSPSpace(
                                            domain_dim = domain_dim, 
                                            length_scale = length_scale,
                                            n_rotates = n_rotates, 
                                            n_scales = n_scales,
                                            scale_min = scale_min, 
                                            scale_max = scale_max
                                            )

        self.size_out = self.rep.ssp_dim
        mapped_low = np.ones( (self.size_out,) ) * -1
        mapped_high = np.ones( (self.size_out,) )
        
        self.observation_space = spaces.Box( mapped_low, mapped_high )
        
        
    def observation(self, observation):
        '''
        Transforms the observation to a HexSSP representation

        Args:
            observation: The observation to transform

        Returns:
            The observation with HexSSP represention
        '''
        
        observation_mapped = self.rep.encode(observation.reshape(1,-1)).reshape(1,-1)
        #print('observation: ', observation, 'mapped: ', observation_mapped)
        
        return observation_mapped