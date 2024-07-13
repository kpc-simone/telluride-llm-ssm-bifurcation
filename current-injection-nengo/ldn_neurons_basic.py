import matplotlib.pyplot as plt
import numpy as np

import nengo
import scipy.linalg
from scipy.special import legendre


# action plan
# LDN in neurons -- DONE :)
# LDN in neurons w/ controllable weights
# LDN in neurons w/ controllable biases

class LDN_RNN(object):
    def __init__(self, q = 6, theta = 1., size_in = 1, n_neurons = 1000, tau_synapse = 0.05, neuron_type = nengo.LIFRate() ):
        '''
        adapted from Terry Stewart's LDN implementation
        https://github.com/tcstewar/testing_notebooks/blob/master/ldn/Basic%20Example.ipynb
        '''
        self.q = q              # number of internal state dimensions per input
        self.theta = theta      # size of time window (in seconds)
        self.size_in = size_in  # number of inputs
        self.n_neurons = n_neurons
        self.tau_synapse = tau_synapse
        self.neuron_type = neuron_type
        
        # Do Aaron Voelker's math to generate the matrices A and B so that
        #  dx/dt = Ax + Bu will convert u into a legendre representation over a window theta
        #  https://github.com/arvoelke/nengolib/blob/master/nengolib/synapses/analog.py#L536
        A = np.zeros((q, q))
        B = np.zeros((q, 1))
        for i in range(q):
            B[i] = (-1.)**i * (2*i+1)
            for j in range(q):
                A[i,j] = (2*i+1)*(-1 if i<j else (-1.)**(i-j+1))
        self.A = A / theta
        self.B = B / theta
        
        self.network = nengo.Network(seed=0)
        with self.network:
            self.input = nengo.Node(size_in = self.size_in)
            self.fw = nengo.Ensemble(
                        n_neurons = self.n_neurons,
                        dimensions = self.size_in,
                        radius = 1.,
                        neuron_type = self.neuron_type
                        )
            
            self.rec = nengo.Ensemble( 
                        n_neurons = self.n_neurons,
                        dimensions = self.q,
                        radius = np.sqrt(2),
                        neuron_type = self.neuron_type
                        )
            
            def forward(x):
                return self.tau_synapse * self.B @ x
            
            def recurrent(x):
                return self.tau_synapse * self.A @ x + x
        
            nengo.Connection( self.input, self.fw, synapse = None )
            nengo.Connection( self.fw, self.rec, function = forward, synapse = self.tau_synapse )
            nengo.Connection( self.rec, self.rec, function = recurrent, synapse = self.tau_synapse )

    def get_weights_for_delays(self, r):
        # compute the weights needed to extract the value at time r
        # from the network (r=0 is right now, r=1 is theta seconds ago)
        r = np.asarray(r)
        m = np.asarray([legendre(i)(2*r - 1) for i in range(self.q)])
        return m.reshape(self.q, -1).T

with nengo.Network() as model:
    q = 11
    
    input = nengo.Node(nengo.processes.WhiteSignal(2., high=5, seed=0))
    ldn_rnn = LDN_RNN( q = q, size_in = 1, theta = 0.5)
    output = nengo.Node(None, size_in=2)
    
    nengo.Connection(input, ldn_rnn.input, synapse=None)

    nengo.Connection(ldn_rnn.rec, output[0], synapse=None, transform=ldn_rnn.get_weights_for_delays([0.5]))
    nengo.Connection(ldn_rnn.rec, output[1], synapse=None, transform=ldn_rnn.get_weights_for_delays([1.0]))
    
    in_p = nengo.Probe(input)
    out_p = nengo.Probe(output)
    
if __name__ == "__main__":
    sim = nengo.Simulator(model)
    sim.run(5.)
    
    fig,axs = plt.subplots(3,1,sharex=True)
    axs[0].plot(sim.trange(),sim.data[in_p],label='Input')
    axs[1].plot(sim.trange(),sim.data[out_p][:,0],label='Delay = 0.5')
    axs[2].plot(sim.trange(),sim.data[out_p][:,1],label='Delay = 1.0')
    for ax in axs.ravel():
        ax.legend(loc='upper left',fancybox=False)
    ax.set_xlabel('Time(s)')
    plt.show()