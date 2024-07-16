import matplotlib.pyplot as plt
import numpy as np
import sys,os

sys.path.insert(0,'.')
from policies import GaussianPolicy

def test_gaussian_policy_sampling():
    action_dim = 3
    gp = GaussianPolicy( action_dim = action_dim )
    
    param_matrix = np.tile(np.atleast_2d([0.,1.5]).T, action_dim )
    param_matrix += np.random.normal(loc=0,scale=0.5,size=param_matrix.shape)
    
    # ensure non-negative variance
    param_matrix[1,:] = np.clip(param_matrix[1,:], a_min = 0.1, a_max = None )

    n_samples = 100000
    samples = np.zeros( (n_samples,action_dim) )
    for i in range(n_samples):
        samples[i,:] = gp.sample( param_matrix ).reshape(1,action_dim)
    
    for d in range(action_dim):
        samples_ = samples[:,d]
        error_mean = param_matrix[0,d] - np.mean(samples_)
        error_std = param_matrix[1,d] - np.std(samples_)
        assert np.abs(error_mean) < 0.01
        assert np.abs(error_std) < 0.01
    
def test_gaussian_policy_gradient():
    action_dim = 3
    gp = GaussianPolicy( action_dim = action_dim )
    
    param_matrix = np.tile(np.atleast_2d([0.,1.5]).T, action_dim )
    param_matrix += np.random.normal(loc=0,scale=0.5,size=param_matrix.shape)
    
    # ensure non-negative variance
    param_matrix[:,1] = np.clip(param_matrix[:,1], a_min = 0.1, a_max = None )

    action = gp.sample( param_matrix )
    print('action: ', action)
    gradient_log_pi = gp.gradient_log_pi( action, param_matrix )
    print('full grad: ', gradient_log_pi)
    
if __name__ == '__main__':
    #test_gaussian_policy_sampling()
    test_gaussian_policy_gradient()