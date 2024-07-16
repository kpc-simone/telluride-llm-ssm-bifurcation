import numpy as np

class GaussianPolicy(BasePolicy):
    # from Sutton & Barto (2018) Section 13.7
    def __init__(self,
        action_dim = 1,
        ):
        
        self.params_per_dim = 2        # mean and standard deviation
        self.action_dim = action_dim
        
    def compute_pmf(self,param_matrix):
        mus = param_matrix[:,0].reshape(-1,1)
        sigmas = ( param_matrix[:,1].reshape(-1,1) )**2
        # print('mus shape: ', mus.shape)
        # print('sigmas shape: ', sigmas.shape)
        
        return mus, sigmas
        
    def sample(self,param_matrix):
        mus, sigmas = self.compute_pmf(param_matrix)
        
        return np.random.normal( loc = mus, scale = sigmas, size = (self.action_dim,1) )
        
    def gradient_log_pi(self,action,param_matrix):
        mus, sigmas = self.compute_pmf(param_matrix)
        
        grad_mu =  ( action - mus ) / ( sigmas**2 )
        grad_sigma = ( action - mus )**2 / ( sigmas**2 )
        # print('grad mu: ', grad_mu)
        # print('grad pi: ', grad_sigma)
        
        return np.clip( np.hstack( [grad_mu,grad_sigma] ), a_min = -10, a_max = 10 )