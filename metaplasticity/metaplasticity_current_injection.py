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

n_neurons = 100

model = nengo.Network()
with model:
    model.config[nengo.Connection].synapse=None
    
    stim = nengo.Node( nengo.processes.WhiteSignal(1.0, high=5, seed=0) )
    
    # change to present input
    inj = nengo.Node( nengo.processes.PresentInput( inputs = [-0.5, 0.5], presentation_time = 1.) )
    
    output = nengo.Node(None, size_in=1)
    ens = nengo.Ensemble(n_neurons = n_neurons, 
                        dimensions = 1,
                        neuron_type = nengo.Tanh(),
                        )
    nengo.Connection(stim, ens)
    nengo.Connection(inj,ens) 
    
    learn = nengo.Node( Learner( 
                                    size_in = ens.n_neurons, 
                                    size_out = stim.size_out,
                                    tau_learn = 0.02)
                                    )
    error = nengo.Node(None, size_in=1)
    nengo.Connection(ens.neurons, learn[:ens.n_neurons])
    
    meta = nengo.Node(lambda t, x:1/(x+1)**3, size_in=n_neurons, size_out=n_neurons)
    nengo.Connection(ens.neurons, meta, synapse=0.1)
    nengo.Connection(meta, learn[ens.n_neurons:ens.n_neurons*2])
    
    nengo.Connection(learn, output)
    
    nengo.Connection(output, error)
    nengo.Connection(stim, error, transform=-1)
    nengo.Connection(error, learn[-stim.size_out:], synapse=0)
    
    p_error = nengo.Probe(error)

if __name__ == "__main__":

    sim = nengo.Simulator(model)
    with sim:
        sim.run(0.5)
        
    plt.figure(figsize=(14,3))
    plt.plot(sim.trange(), np.linalg.norm(sim.data[p_error], axis=1), lw=1, label='error')
    plt.legend()
    plt.show()
