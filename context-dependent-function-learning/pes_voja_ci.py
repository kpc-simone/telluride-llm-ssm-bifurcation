
import pandas as pd
import numpy as np
import nengo

model = nengo.Network()
n_neurons = 200

rep_learn_time = 150.
sim_run_time = 2*rep_learn_time
voja_on = False
learning_rate = 0.0000005

with model:
    stim = nengo.Node( nengo.processes.WhiteSignal(1.0, high=5, seed=0) )
    
    # change to present input
    inj = nengo.Node( nengo.processes.PresentInput( inputs = [0., 0.5], presentation_time = 1.) )
    
    ens = nengo.Ensemble( 
            n_neurons = n_neurons,
            dimensions = 1,
            intercepts = np.clip(np.random.beta(a=0.2,b=0.2,size=(n_neurons,)),a_min=0,a_max=1),
            neuron_type = nengo.Tanh(),
            encoders = nengo.dists.Choice([[1]])
            )
    out = nengo.Ensemble(
            n_neurons = n_neurons,
            dimensions = 1,
            neuron_type = nengo.Tanh(),
        )
    
    nengo.Connection(stim,ens)
    
    output = nengo.Node(None,size_in=1)

    if voja_on == True:
        conn1 = nengo.Connection(ens,out,
                        learning_rule_type = nengo.Voja(learning_rate = 0.0001)
                        )
        rep_learning_control = nengo.Node( nengo.processes.PresentInput( inputs = [1., 0.], presentation_time = rep_learn_time) )
        nengo.Connection(rep_learning_control, conn1.learning_rule, synapse = None)
        nengo.Connection(inj,ens)        
    else:
        nengo.Connection(ens,out)
        nengo.Connection(inj,out)
        
    # transform is a weight matrix; essentially a dot product
    # to implement the PES learning rule
    conn2 = nengo.Connection(out.neurons,output,
                        # initialize
                        transform = np.zeros((1,n_neurons)),
                        learning_rule_type=nengo.PES(learning_rate = learning_rate)
                        )
    
    error = nengo.Node(None,size_in=1)
    nengo.Connection(output,error)
    nengo.Connection(stim,error,transform=-1)
    nengo.Connection(error,conn2.learning_rule)
    
    err_p = nengo.Probe(error)
    
if __name__ == '__main__':
    with nengo.Simulator(model) as sim:
        sim.run(sim_run_time)
    
    df = pd.DataFrame( data = {
            'time'  : sim.trange(),
            'error' : sim.data[err_p].flatten()
            }
        )
    df.to_csv(f'rep_learning-{voja_on}.csv')
        
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(1,1)
    ax.plot(df['time'],df['error'].pow(2).rolling(10000,min_periods = 0).apply(lambda x: np.sqrt(x.mean())),color='dimgray')
    plt.show()