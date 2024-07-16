from scipy.special import softmax
import numpy as np

class BasePolicy(object):
    def compute_pmf(self,param_vector):
        pass
        
    def sample(self,param_vector):
        pass
    
    def gradient_log_pi(self,action,param_vector):
        pass