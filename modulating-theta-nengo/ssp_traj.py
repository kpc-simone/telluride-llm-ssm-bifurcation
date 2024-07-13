import nengo
import numpy as np
from sspspace import  RandomSSPSpace, SSPSpace
import base64

from io import BytesIO as StringIO
import PIL
from PIL import Image

# Create SSP space
domain_dim = 1
d = 151
x_scale = 0.03
time_scale = 0.03
length_scales = np.array([[time_scale],[x_scale]])
radius = 1
seed = 0
T= 5
dt = 0.001

def gram_schmidt(A):
    d, n = A.shape
    Q = np.zeros((d, n))
    for i in range(n):
        q = A[:, i]
        for j in range(i):
            q -= np.dot(Q[:, j], A[:, i]) * Q[:, j]
        q /= np.linalg.norm(q)
        Q[:, i] = q
    return Q

space_bounds = radius * np.tile([-1, 1], (domain_dim, 1))
time_bounds = np.array([[0,T]])
bounds = np.vstack([time_bounds, space_bounds])

tssp_space = RandomSSPSpace(domain_dim + 1, ssp_dim=d, domain_bounds=bounds, 
                                length_scale=length_scales, seed=seed)
tssp_space.phase_matrix = gram_schmidt(tssp_space.phase_matrix)

d = tssp_space.ssp_dim
ssp_space = SSPSpace(domain_dim, ssp_dim=d, domain_bounds=space_bounds, 
                            length_scale=x_scale, phase_matrix = tssp_space.phase_matrix[:,1:])  
t_space = SSPSpace(1, ssp_dim=d, domain_bounds=time_bounds, 
                            length_scale=time_scale, phase_matrix = tssp_space.phase_matrix[:,0].reshape(-1,1))                         

def get_to_Fourier(d):
    k = (d+1)//2
    M = np.zeros((2*k ,d))
    M[:-1:2,:] = np.fft.fft(np.eye(d))[:k,:].real
    M[1::2,:] = np.fft.fft(np.eye(d))[:k:].imag
    return M

def get_from_Fourier(d):
    k = (d+1) // 2 

    shiftmat = np.zeros((2*d, 2*k))
    shiftmat[:k,::2] = np.eye(k)
    shiftmat[k:2*k-1,2::2] = np.flip(np.eye(k-1), axis=0)
    shiftmat[2*k-1:3*k-1,1::2] = np.eye(k)
    shiftmat[3*k-1:,3::2] = -np.flip(np.eye(k-1), axis=0)

    invW = np.fft.ifft(np.eye(d))
    M = np.hstack([invW.real, - invW.imag]) @ shiftmat 
    return M

to_Fourier = get_to_Fourier(d)
to_SSP = get_from_Fourier(d)

class PlotSSP(nengo.Node):
    def __init__(self, ssp_space, x_vals, y_vals):
        self.ssp_space = ssp_space
        self.d = ssp_space.ssp_dim
        self.recalc_decoder(x_vals, y_vals)
        self.init_x_vals = x_vals
        self.init_y_vals = y_vals

        template = '''
            <svg width="100%%" height="100%%" viewbox="0 0 100 100">
                {img}
                <text x="5" y="50" style="fill:yellow;font-size:5px">t-{x_min}</text>
                <text x="95" y="50" text-anchor="end" style="fill:yellow;font-size:5px">t</text>
                <text x="50" y="5" text-anchor="middle" style="fill:yellow;font-size:5px">{y_min}</text>
                <text x="50" y="95" text-anchor="middle" style="fill:yellow;font-size:5px">{y_max}</text>
            </svg>'''
            

        def plot(t, x):
            y = np.flip(self.decoder @ x, axis=0)
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

        super().__init__(plot, size_in=self.d, size_out=0)
        self.output._nengo_html_ = template.format(img='', x_min=self.x_vals[0],
                                                           x_max=self.x_vals[-1],
                                                           y_min=self.y_vals[0],
                                                           y_max=self.y_vals[-1])

    def recalc_decoder(self, x_vals, y_vals):
        self.x_vals = x_vals
        self.y_vals = y_vals
        Xs = np.meshgrid(x_vals,y_vals)
        grid_pts = np.vstack([Xs[i].reshape(-1) for i in range(2)]).T
        self.decoder = self.ssp_space.encode(grid_pts).reshape(len(x_vals), len(y_vals), self.d)

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
        if 'z' in keys_pressed:
            plt.recalc_decoder(x_vals=self.init_x_vals, y_vals=self.init_y_vals)        


input_scaling = 1#0.5
max_rad = T*input_scaling
n_neurons = 800
leak = 0.999
tau = 0.1
model = nengo.Network(seed=seed)
with model:
    t_scale = 10
    
    stim_x = nengo.Node(nengo.processes.WhiteSignal(20, high=.3, seed=3), size_out=domain_dim)
    stim = nengo.Node(lambda t, x: ssp_space.encode(x).flatten(), size_in=domain_dim)
    nengo.Connection(stim_x, stim)
    
    memory = nengo.networks.EnsembleArray(n_neurons, (d+1)//2, ens_dimensions=2, radius=max_rad)
    nengo.Connection(stim, memory.input, transform=input_scaling*tau*to_Fourier, synapse=tau)
    for i in range((d+1)//2):
        ens = memory.ea_ensembles[i]
        freq = -t_space.phase_matrix[i,0]/time_scale
        nengo.Connection(ens, ens, transform = np.array([[leak,-tau*freq],[tau*freq,leak]]).T, synapse=tau)
        
 
    output = nengo.Node(size_in=d)
    nengo.Connection(memory.output, output, synapse=tau, transform=to_SSP)
    
    tssp_plt = PlotSSP(tssp_space, x_vals=np.linspace(T, 0, 101),
                        y_vals=np.linspace(-radius*1.2, radius*1.2, 101))
    nengo.Connection(output, tssp_plt)

def on_step(sim):
    tssp_plt.move_plot(__page__.keys_pressed)