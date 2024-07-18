import matplotlib.pyplot as plt
import numpy as np
import nengo

class Learner(nengo.processes.Process):
    def __init__(self, size_in, size_out, tau_learn):
        self.size_in = size_in
        self.size_out = size_out
        self.tau_learn = tau_learn   # convergence time of the learning process
        super().__init__(default_size_in=size_in*2+size_out,
                         default_size_out=size_out)
        
    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        w = np.zeros((self.size_out, self.size_in))
        learn_scale = 1-np.exp(-dt/self.tau_learn)
        
        def step_learn(t, x, w=w, learn_scale=learn_scale):
            pre, meta, error = x[:self.size_in], x[self.size_in:self.size_in*2],x[self.size_in*2:]
            
            # compute the learning rate that will give the desired convergence time
            lr = np.sum(pre*pre*meta)
            if lr != 0:
                lr = 1.0 / lr
            lr = learn_scale * lr
            
            w -= lr * np.outer(error, pre*meta)            
            
            return w @ pre
        return step_learn

D = 64
N = 500
n_pairs = 3

import nengo_spa as spa
vocab = spa.Vocabulary(D, strict=False)
pairs = [(vocab.parse(f'A{i}'), vocab.parse(f'B{i}')) for i in range(n_pairs)]

t_present = 0.1


model = nengo.Network()
with model:
    model.config[nengo.Connection].synapse=None
    stim = nengo.Node(nengo.processes.PresentInput([p[0].v for p in pairs], presentation_time=t_present))
    target = nengo.Node(nengo.processes.PresentInput([p[1].v for p in pairs], presentation_time=t_present))
    output = nengo.Node(None, size_in=D)
    ens = nengo.Ensemble(n_neurons=N, dimensions=D, neuron_type=nengo.LIFRate(),
                        )
    nengo.Connection(stim, ens)
    
    learn = nengo.Node( Learner( 
                                    size_in = ens.n_neurons, 
                                    size_out = target.size_out,
                                    tau_learn = 0.02)
                                    )
    error = nengo.Node(None, size_in=D)
    nengo.Connection(ens.neurons, learn[:ens.n_neurons])
    
    importance = nengo.Node(None, size_in=N)
    
    # compute derivative -- newly spiking neurons will be preferentially updated
    nengo.Connection(ens.neurons, importance, synapse=0.002)
    nengo.Connection(ens.neurons, importance, synapse=0.02, transform=-1)
    nengo.Connection(importance, learn[ens.n_neurons:ens.n_neurons*2])
    
    nengo.Connection(learn, output)
    
    
    nengo.Connection(output, error)
    nengo.Connection(target, error, transform=-1)
    nengo.Connection(error, learn[-target.size_out:], synapse=0)
    
    p_error = nengo.Probe(error)

if __name__ == "__main__":

    sim = nengo.Simulator(model)
    with sim:
        sim.run(0.5)
        
    plt.figure(figsize=(14,3))
    plt.plot(sim.trange(), np.linalg.norm(sim.data[p_error], axis=1), lw=1, label='error')
    plt.legend()
    plt.show()
