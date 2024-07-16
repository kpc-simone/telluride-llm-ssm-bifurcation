from base_policy import BasePolicy

from scipy.special import softmax
import numpy as np

class SoftmaxPolicy(BasePolicy):
    def __init__(self, 
        n_params = 3,
        temperature = 2., 
        ):
        
        # for discrete action spaces, n_params = number of actions
        self.n_params = n_params
        self.temperature = temperature

    def compute_pmf(self,param_vector):
        return softmax( param_vector / self.temperature )
    
    def sample(self,param_vector):
        ps = self.compute_pmf(param_vector)
        sample = np.random.choice( range(self.n_params), p = ps.flatten() )

        return sample
        
    def gradient_log_pi(self,action,param_vector):
        expected_vector = self.compute_pmf(param_vector)
        
        sample_vector = np.zeros(self.n_params)
        sample_vector[action] = 1
        
        return sample_vector - expected_vector