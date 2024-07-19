import matplotlib.pyplot as plt
import numpy as np
import nengo

t_start = 0
t_end = 20
        
n_neurons = 200

tau_learn = 0.05
window_size = 100

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
            
            if t < t_start:
                error = 0
                
            if t > t_end:
                error = 0
            
            # compute the learning rate that will give the desired convergence time
            lr = np.sum(pre*pre*meta)
            if lr != 0:
                lr = 1.0 / lr
            lr = learn_scale * lr
            
            w -= lr * np.outer(error, pre*meta)            
            
            return w @ pre
        return step_learn
        
class WindowedFunc(object):
    def __init__(self,size_in,size_out,function='variance',window_size=window_size):
        self.size_in = size_in
        self.size_out = size_out
        self.function = function
        self.history = np.zeros((window_size,self.size_out))
    
    def __call__(self,t,x):
        self.history = np.roll(self.history, shift=-1, axis=0)
        self.history[-1,:] = x
        
        return np.std(self.history,axis=0)

model = nengo.Network()
with model:
    model.config[nengo.Connection].synapse=None
    
    stim = nengo.Node( nengo.processes.WhiteSignal( 1.0, high = 5, seed = 0 ) )
    
    # change to present input
    inj = nengo.Node( nengo.processes.PresentInput( inputs = [-0.5, 0.5], presentation_time = 10.) )
    
    output = nengo.Node( None, size_in = 1 )
    ens = nengo.Ensemble(n_neurons = n_neurons, 
                        dimensions = 1,
                        neuron_type = nengo.LIFRate(),
                        intercepts = np.random.beta(a=0.2,b=0.2,size=(n_neurons,)),
                        #max_rates = 100*np.ones((n_neurons,))
                        )
    nengo.Connection( stim, ens)
    nengo.Connection( inj, ens.neurons,transform=np.ones((n_neurons,1))) 
    
    learn = nengo.Node( Learner( 
                                    size_in = ens.n_neurons, 
                                    size_out = stim.size_out,
                                    tau_learn = tau_learn)
                                    )
                                    
    error = nengo.Node( None, size_in = 1 )
    nengo.Connection(ens.neurons, learn[:ens.n_neurons])
    
    meta = nengo.Node( WindowedFunc(size_in=n_neurons,size_out=n_neurons),size_in = n_neurons )
    nengo.Connection(ens.neurons, meta, synapse=None)
    nengo.Connection(meta, learn[ens.n_neurons:ens.n_neurons*2])
    
    nengo.Connection(learn, output)
    
    nengo.Connection(output, error)
    nengo.Connection(stim, error, transform=-1)
    nengo.Connection(error, learn[-stim.size_out:], synapse=0)
    
    p_error = nengo.Probe(error)
    p_out = nengo.Probe(output)
    p_stim = nengo.Probe(stim)

import pandas as pd

if __name__ == "__main__":

    with nengo.Simulator(model) as sim:
        sim.run(30.)
        
    df = pd.DataFrame( data = {
            'time'      : sim.trange(),
            'error'     : sim.data[p_error].flatten(),
            'stim'      : sim.data[p_stim].flatten(),
            'output'    : sim.data[p_out].flatten()
            } 
        )
    df.to_csv(f'expt_tstart{t_start}_tend{t_end}_taulearn{tau_learn}_winsize{window_size}.csv')
    
    plt.figure(figsize=(14,3))
    plt.plot(sim.trange(), np.linalg.norm(sim.data[p_error], axis=1), lw=1, label='error')
    plt.legend()
    plt.show()
