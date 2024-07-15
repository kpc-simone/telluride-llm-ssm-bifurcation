import matplotlib.pyplot as plt
import numpy as np

import nengo
import scipy.linalg
from scipy.special import legendre

from ldn_neurons_basic import LDN_RNN

with nengo.Network() as model:
    q = 11
    theta = 0.5
    
    input = nengo.Node(nengo.processes.WhiteSignal(5., high=5, seed=1))
    ldn_rnn = LDN_RNN( q = q, size_in = 1, theta = theta, neuron_type = nengo.LIFRate() )
    output = nengo.Node(None, size_in=1)
    
    nengo.Connection(input, ldn_rnn.input, synapse=None)
    nengo.Connection(ldn_rnn.rec, output, synapse=None, transform=ldn_rnn.get_weights_for_delays([1.0]))
    
    in_p = nengo.Probe(input)
    out_p = nengo.Probe(output)
    
    # inject current into recurrent population to mimic loss of inhibition
    inj = nengo.Node( lambda t: t//1./10 if t<10 else 1.-(t-10.)//1./10)
    nengo.Connection(inj,ldn_rnn.rec.neurons,transform = np.ones((1000,1)) )
    inj_p = nengo.Probe(inj)
    
if __name__ == "__main__":
    sim = nengo.Simulator(model)
    sim.run(30.)
    
    fig,axs = plt.subplots(3,1,sharex=True)
    axs[0].plot(sim.trange(),sim.data[in_p],label='Input')
    axs[0].plot(sim.trange()+0.5,sim.data[in_p],label='Input (delayed)')
    axs[1].plot(sim.trange(),sim.data[inj_p],label='Current injected')
    axs[2].plot(sim.trange(),sim.data[out_p],label='Delay = 1.0')
    for ax in axs.ravel():
        ax.legend(loc='upper left',fancybox=False)
        for t in np.linspace( 0, sim.trange().max(), 31):
            ax.axvline(t,color='k',linestyle='--')
    ax.set_xlabel('Time(s)')
    plt.show()