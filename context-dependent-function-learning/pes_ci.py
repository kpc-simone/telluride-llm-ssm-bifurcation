import numpy as np
import nengo

model = nengo.Network()

n_neurons = 200

with model:
    stim = nengo.Node( nengo.processes.WhiteSignal(1.0, high=5, seed=0) )
    
    # change to present input
    inj = nengo.Node([0.])
    
    ens = nengo.Ensemble( 
            n_neurons = n_neurons,
            dimensions = 1,
            intercepts = np.clip(np.random.beta(a=0.2,b=0.2,size=(n_neurons,)),a_min=0,a_max=1),
            neuron_type = nengo.Tanh(),
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
    
    error = nengo.Node(None,size_in=1)
    nengo.Connection(output,error)
    nengo.Connection(stim,error,transform=-1)
    nengo.Connection(error,conn.learning_rule)
    
    err_p = nengo.Probe(error)
    
if __name__ == '__main__':
    with nengo.Simulator(model) as sim:
        sim.run(120.)
        
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(1,1)
    ax.plot(sim.trange(),sim.data[err_p])
    plt.show()