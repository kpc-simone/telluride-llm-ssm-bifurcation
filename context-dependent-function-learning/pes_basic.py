import numpy as np
import nengo

model = nengo.Network()

with model:
    stim = nengo.Node( nengo.processes.WhiteSignal(1.0, high=5, seed=0) )
    
    ens = nengo.Ensemble( 
            n_neurons = 100,
            dimensions = 1,
            )
    nengo.Connection(stim,ens)
    
    output = nengo.Node(None,size_in=1)
    
    # transform is a weight matrix; essentially a dot product
    # to implement the PES learning rule
    conn = nengo.Connection(ens.neurons,output,
                        # initialize
                        transform = np.zeros((1,100)),
                        learning_rule_type=nengo.PES(learning_rate = 0.00001)
                        )
    
    error = nengo.Node(None,size_in=1)
    nengo.Connection(output,error)
    nengo.Connection(stim,error,transform=-1)
    nengo.Connection(error,conn.learning_rule)
    
    