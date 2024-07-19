import numpy as np
import nengo

model = nengo.Network()

n_neurons = 200

with model:
    stim = nengo.Node( nengo.processes.WhiteSignal(1.0, high=5, seed=0) )
    
    # change to present input
    inj = nengo.Node( nengo.processes.PresentInput([-0.5,0.5],presentation_time=10.) )
    
    ens = nengo.Ensemble( 
            n_neurons = n_neurons,
            dimensions = 1,
            intercepts = np.clip(np.random.beta(a=0.2,b=0.2,size=(n_neurons,)),a_min=0,a_max=1),
            neuron_type = nengo.LIFRate(),
            )
    nengo.Connection(stim,ens)
    nengo.Connection(inj,ens)
    
    output = nengo.Node(None,size_in=1)
    
    # transform is a weight matrix; essentially a dot product
    # to implement the PES learning rule
    conn = nengo.Connection(ens.neurons,output,
                        # initialize
                        transform = np.zeros((1,n_neurons)),
                        learning_rule_type=nengo.PES(learning_rate = 0.00001)
                        )
    
    error = nengo.Node( lambda t,x: x if t < 20 else 0, size_in = 1 )
    nengo.Connection(output,error)
    nengo.Connection(stim,error,transform=-1)
    nengo.Connection(error,conn.learning_rule)
    
    p_error = nengo.Probe(error)
    p_stim = nengo.Probe(stim)
    p_out = nengo.Probe(output)
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    
    with nengo.Simulator(model) as sim:
        sim.run(30.)
        
    df = pd.DataFrame( data = {
            'time'      : sim.trange(),
            'error'     : sim.data[p_out].flatten() - sim.data[p_stim].flatten(),
            'stim'      : sim.data[p_stim].flatten(),
            'output'    : sim.data[p_out].flatten()
            } 
        )
    df.to_csv('pes_baseline.csv')

    plt.figure(figsize=(14,3))
    plt.plot(df['time'], df['error'], lw=1, label='error')
    plt.legend()
    plt.show()