import nengo
import nengo_spa as spa
import numpy as np

import base64
from io import BytesIO as StringIO
import PIL
from PIL import Image
class SpikePlot(nengo.Node):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.mapping = np.arange(width*height)

        template = '''
            <svg width="100%%" height="100%%" viewbox="0 0 100 100">
                {img}
            </svg>'''
            

        def plot(t, x):
            #x = np.clip(x * 255, 0, 255)
            x = np.clip(x * 128 + 128, 0, 255)
            x = x.astype('uint8')
            x = x[self.mapping]
            x.shape = (self.height, self.width)

            png = Image.fromarray(x[:,:])
            buffer = StringIO()
            png.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                                
            img = '''<image width="100%%" height="100%%"
                      xlink:href="data:image/png;base64,%s" 
                      style="image-rendering: pixelated;"/>
                  ''' % img_str

            plot._nengo_html_ = template.format(img=img)

        super().__init__(plot, size_in=width*height, size_out=0)
        self.output._nengo_html_ = template.format(img='')
        
    def score(self, i, enc):
        N = self.width*self.height
        centre = enc[self.mapping[i]]
        others = enc[[self.mapping[(i-1)%N], 
                      self.mapping[(i+1)%N],
                      self.mapping[(i+self.width)%N],
                      self.mapping[(i-self.width)%N],
                    ]]
        score = np.dot(others, centre)
        return np.sum(score)

    def optimize(self, sim, obj):
        enc = np.vstack([sim.data[ens].encoders for ens in obj.all_ensembles])
        print(enc.shape)

        indices = np.random.randint(0, self.width*self.height, size=(100,2))
        for i,j in indices:
            if i == j: return
            score1 = self.score(i, enc) + self.score(j, enc)
            self.mapping[[i,j]] = self.mapping[[j,i]]
            score2 = self.score(i, enc) + self.score(j, enc)
            if score1 > score2:
                self.mapping[[i,j]] = self.mapping[[j,i]]


use_hex = False
if use_hex:
    import grid_cells
    basis = grid_cells.GridBasis(dimensions=2, n_rotates=12, scales=np.linspace(0.25, 4, 16))
    D = basis.axes.shape[1]
    vocab = spa.Vocabulary(D)
    vocab.add('X', basis.axes[0])
    vocab.add('T', basis.axes[1])
else:
    D = 256
    vocab = spa.Vocabulary(D)
    vocab.add('X', vocab.algebra.create_vector(D, {"positive", "unitary"}))
    vocab.add('T', vocab.algebra.create_vector(D, {"positive", "unitary"}))
X = vocab.parse('X')
T = vocab.parse('T')
lnT = spa.SemanticPointer(np.fft.ifft(np.log(np.fft.fft(T.v))).real)


import nengo
import numpy as np
import base64

from io import BytesIO as StringIO
import PIL
from PIL import Image
class PlotSSP(nengo.Node):
    def __init__(self, X, Y, x_vals, y_vals):
        self.X = X
        self.Y = Y
        self.recalc_decoder(x_vals, y_vals)

        template = '''
            <svg width="100%%" height="100%%" viewbox="0 0 100 100">
                {img}
                <text x="5" y="50" style="fill:yellow;font-size:5px">{x_min}</text>
                <text x="95" y="50" text-anchor="end" style="fill:yellow;font-size:5px">{x_max}</text>
                <text x="50" y="5" text-anchor="middle" style="fill:yellow;font-size:5px">{y_min}</text>
                <text x="50" y="95" text-anchor="middle" style="fill:yellow;font-size:5px">{y_max}</text>
            </svg>'''
            

        def plot(t, x):
            y = np.dot(self.decoder, x)
            y = np.clip(y * 255, 0, 255)
            y = y.astype('uint8')

            png = Image.fromarray(y[:,:])
            buffer = StringIO()
            png.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                                
            img = '''<image width="100%%" height="100%%"
                      xlink:href="data:image/png;base64,%s" 
                      style="image-rendering: pixelated;"/>
                  ''' % img_str

            plot._nengo_html_ = template.format(img=img, x_min=self.x_vals[0],
                                                           x_max=self.x_vals[-1],
                                                           y_min=self.y_vals[0],
                                                           y_max=self.y_vals[-1])

        super().__init__(plot, size_in=len(X), size_out=0)
        self.output._nengo_html_ = template.format(img='', x_min=self.x_vals[0],
                                                           x_max=self.x_vals[-1],
                                                           y_min=self.y_vals[0],
                                                           y_max=self.y_vals[-1])

    def recalc_decoder(self, x_vals, y_vals):
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.decoder = np.zeros((len(y_vals), len(x_vals), len(self.X)))
        Xs = [self.X**x for x in x_vals]
        Ys = [self.Y**y for y in y_vals]
        for i, x in enumerate(Xs):
            for j, y in enumerate(Ys):
                self.decoder[j,i,:] = (x*y).v

    def move_plot(self, keys_pressed):
        if 'w' in keys_pressed:
            S = plt.y_vals[-1]-plt.y_vals[0]        
            plt.recalc_decoder(x_vals=plt.x_vals, y_vals=plt.y_vals-S/4)
        if 's' in keys_pressed:
            S = plt.y_vals[-1]-plt.y_vals[0]        
            plt.recalc_decoder(x_vals=plt.x_vals, y_vals=plt.y_vals+S/4)
        if 'a' in keys_pressed:
            S = plt.x_vals[-1]-plt.x_vals[0]        
            plt.recalc_decoder(x_vals=plt.x_vals-S/4, y_vals=plt.y_vals)
        if 'd' in keys_pressed:
            S = plt.x_vals[-1]-plt.x_vals[0]        
            plt.recalc_decoder(x_vals=plt.x_vals+S/4, y_vals=plt.y_vals)
        if 'q' in keys_pressed:
            Xs = plt.x_vals*2
            Ys = plt.y_vals*2
            Xs += -Xs[len(Xs)//2] + plt.x_vals[len(Xs)//2]
            Ys += -Ys[len(Ys)//2] + plt.y_vals[len(Ys)//2]
            plt.recalc_decoder(x_vals=Xs, y_vals=Ys)
        if 'e' in keys_pressed:
            Xs = plt.x_vals/2
            Ys = plt.y_vals/2
            Xs += -Xs[len(Xs)//2] + plt.x_vals[len(Xs)//2]
            Ys += -Ys[len(Ys)//2] + plt.y_vals[len(Ys)//2]
            plt.recalc_decoder(x_vals=Xs, y_vals=Ys)
        if 'z' in keys_pressed:
            plt.recalc_decoder(x_vals=np.linspace(-5,5,R), y_vals=np.linspace(-5,5,R))        


R = 101
model = spa.Network()
with model:
    t_scale = 10
    
    stim_x = nengo.Node(lambda t: np.sin(t*2*np.pi))
    stim = nengo.Node(lambda t, x: (X**x[0]).v, size_in=1)
    nengo.Connection(stim_x, stim)
    
    state = spa.State(D, neurons_per_dimension=100, subdimensions=32,
                      represent_cc_identity=False)
    for ens in state.all_ensembles:
        ens.intercepts=nengo.dists.Uniform(0,0.05)



    #for ens in state.all_ensembles:
    #    ens.neuron_type=nengo.Direct()
    tau = 0.1
    
    nengo.Connection(stim, state.input, synapse=tau, transform=tau*t_scale/2)
    
    M = (lnT).get_binding_matrix()*t_scale
    nengo.Connection(state.output, state.input, 
                     transform=tau*M+np.eye(D), synapse=tau)
    
    


    plt = PlotSSP(T, X, x_vals=np.linspace(10, 0, R),
                        y_vals=np.linspace(3, -3, R))
    nengo.Connection(state.output, plt)
    
    
    spike_plot = SpikePlot(256,100)
    for i,ens in enumerate(state.all_ensembles):
        nengo.Connection(ens.neurons, spike_plot[i*ens.n_neurons:(i+1)*ens.n_neurons], 
                         transform=0.03, synapse=0.1)
        nengo.Connection(ens.neurons, spike_plot[i*ens.n_neurons:(i+1)*ens.n_neurons], 
                         transform=-0.03, synapse=0.5)
    
    

def on_step(sim):
    plt.move_plot(__page__.keys_pressed)
    if 'r' in __page__.keys_pressed:
        spike_plot.optimize(sim, state)