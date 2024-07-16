from scipy.special import log_softmax
import scipy.special
import numpy as np
import nengo
import math

def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))

## Convert sparsity parameter to neuron bias/intercept
def sparsity_to_x_intercept(d, p):
    sign = 1
    if p > 0.5:
        p = 1.0 - p
        sign = -1
    return sign * np.sqrt(1-scipy.special.betaincinv((d-1)/2.0, 0.5, 2*p))

def ReLU( scaled_encoders,bias,test_phis ):
    return np.clip( np.dot(test_phis, scaled_encoders) + bias, a_min = 0., a_max = None )