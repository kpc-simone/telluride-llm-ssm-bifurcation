import matplotlib.pyplot as plt
import numpy as np

import nengo
import scipy.linalg
from scipy.special import legendre
from scipy.linalg import norm, eigh

def generate_diagonalizable_matrix(n, norm_bound=1):
    # Generate a random orthogonal matrix Q
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    
    # Generate a random diagonal matrix D with bounded norm
    D = np.diag(np.random.uniform(-1, 1, n))
    
    # Construct the diagonalizable matrix A = QDQ^T
    A = Q @ D @ Q.T
    
    return norm_bound*A/norm(A)
    
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
        
        self.C1 = generate_diagonalizable_matrix( self.q )
        self.C2 = generate_diagonalizable_matrix( self.q )
        self.r_mats = { -1: self.C1, 0: np.eye(self.q), 1: self.C2  }
        self.keys_list = list(self.r_mats.keys())

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
                        dimensions = self.q + 1,
                        radius = np.sqrt(2),
                        neuron_type = self.neuron_type
                        )

            def forward(x):
                return self.tau_synapse * self.B @ x

            def recurrent(x_):
                x = x_[:-1]

                idx = np.argmin( [np.abs( x_[-1]-k) for k in self.keys_list ] )
                key = self.keys_list[idx]
                R_Mat = self.r_mats[ key ]
                
                dx = self.tau_synapse * R_Mat @ self.A @ x + x
                dx = np.append(dx,0)
                return dx

            nengo.Connection( self.input, self.fw, synapse = None )
            nengo.Connection( self.fw, self.rec[:-1], function = forward, synapse = self.tau_synapse )
            nengo.Connection( self.rec, self.rec, function = recurrent, synapse = self.tau_synapse )

    def get_weights_for_delays(self, r):
        # compute the weights needed to extract the value at time r
        # from the network (r=0 is right now, r=1 is theta seconds ago)

        r = np.asarray(r)
        m = np.asarray([legendre(i)(2*r - 1) for i in range(self.q)])

        return m.reshape(self.q, -1).T

with nengo.Network() as model:
    q = 3
    theta = 3.

    input = nengo.Node(nengo.processes.WhiteSignal(10, high=0.5, seed=0))
    ldn_rnn = LDN_RNN( q = q, size_in = 1, theta = theta)
    output = nengo.Node(None, size_in=1)

    nengo.Connection(input, ldn_rnn.input, synapse=None)
    nengo.Connection(ldn_rnn.rec[:-1], output, synapse=None, transform=ldn_rnn.get_weights_for_delays([1.]))

    chase = nengo.Node([0])
    nengo.Connection(chase,ldn_rnn.rec[-1],synapse=None)
    
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