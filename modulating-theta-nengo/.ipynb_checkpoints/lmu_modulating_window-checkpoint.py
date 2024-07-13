import matplotlib.pyplot as plt
import nengo
import numpy as np
from lmu_networks import LMUProcess, LMUModulatedProcess, LMUNetwork, LMUModulatedNetwork

theta_start = 0.1
theta_end = 0.3
T = 1.5
dt = 0.001
theta_rate = (theta_start/theta_end)**(1/(1 + int(T/dt))) 
q=8
n_neurons=800
tau=0.03
prb_syn=0.01
model = nengo.Network()
with model:
    inp = nengo.Node(nengo.processes.WhiteSignal(2, high=5, rms=0.3, seed=1))
    inp_ens = nengo.Ensemble(100, 1)
    nengo.Connection(inp, inp_ens, synapse=None)
    lmu = LMUModulatedNetwork(inp_ens, n_neurons, theta=theta_start, q=q, size_in=1, tau=tau)
    
    modulator = nengo.Node(theta_rate)
    nengo.Connection(modulator, lmu.modulator, synapse=None)
    
    recall = nengo.Node(size_in=1)
    nengo.Connection(lmu.output, recall, transform= lmu.get_weights_for_delays(1),  synapse=prb_syn)
    