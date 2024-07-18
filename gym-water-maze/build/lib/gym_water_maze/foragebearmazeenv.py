import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import random

import gymnasium as gym
from gymnasium import spaces

import scipy.linalg
from scipy.linalg import norm, eigh

def generate_diagonalizable_matrix(n, norm_bound=1):
    # Generate a random orthogonal matrix Q
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    
    # Generate a random diagonal matrix D with bounded norm
    D = np.diag(np.random.uniform(-1, 1, n))
    
    # Construct the diagonalizable matrix A = QDQ^T
    A = Q @ D @ Q.T
    
    return norm_bound*A/norm(A)


class LMU():
    def __init__(self, theta, q, size_in=1):
        self.q = q              # number of internal state dimensions per input
        self.theta = theta      # size of time window (in seconds)
        self.size_in = size_in

        # Do Aaron's math to generate the matrices
        #  https://github.com/arvoelke/nengolib/blob/master/nengolib/synapses/analog.py#L536
        Q = np.arange(q, dtype=np.float64)
        R = (2*Q + 1)[:, None] / theta
        j, i = np.meshgrid(Q, Q)

        self.A = np.where(i < j, -1, (-1.)**(i-j+1)) * R
        self.B = (-1.)**Q[:, None] * R

        # discretize A, B
        self.Ad = scipy.linalg.expm(self.A)
        self.Bd = np.dot(np.dot(np.linalg.inv(self.A), (self.Ad-np.eye(self.q))), self.B)

        self.state = np.zeros((self.q, self.size_in))

        self.C1 = generate_diagonalizable_matrix(self.q)
        self.C2 = generate_diagonalizable_matrix(self.q)
        self.C = {True: self.C1, False: self.C2}
            
        super().__init__()

    def step(self, x, reset=False, chase=True):

        if reset:
            self.state = np.dot(self.Bd, np.atleast_2d(x))
        else:
            self.state = np.dot( self.C[chase] @ self.Ad, self.state) + np.dot(self.Bd, np.atleast_2d(x))
        # return self.state.flatten()

    def reset(self):
        self.state = np.zeros((self.q, self.size_in))
        # return self.state.flatten()


# adapted from https://github.com/rpinsler/gym-maze/blob/master/gym_maze/envs/maze.py
class ForageBearMazeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self,
                 radius = 20,
                 goal_radius = 1,
                 action_radius = 1,
                 n_forage_spots = 2,
                 bear_radius = 1,
                 bear_penalty = -10,
                 bear_speed = 0.01,
                 agent_size=0.5,
                 agent_view_radius = 5,
                 dt= 0.001,
                 start = 'S', #S,E,W,N
                 max_steps = 100,
                 render_mode = 'human',
                 reward_type = 'sparse', #sparse, active
                 live_display=False,
                 render_trace=False, penalty=0):
        """Initialize the maze. DType: list"""
        self.max_steps = max_steps
        self.radius = radius
        self.penalty=penalty
        self.n_forage_spots = n_forage_spots
        if n_forage_spots==1:
            self.reward_base_probs = np.array([0.8])
        elif n_forage_spots==2:
            self.reward_base_probs = np.array([0.1,0.4])
        else:
            self.reward_base_probs = np.random.rand(n_forage_spots)
        self.goal_radius = goal_radius
        self.dt = dt
        self.reward_times = np.zeros(n_forage_spots)
        self.reward_probs = np.zeros(n_forage_spots)
        if start=='S':
            self.init_state = np.array([0.,-radius*0.8])
        elif start=='N':
            self.init_state = np.array([0.,radius*0.8])
        elif start=='W':
            self.init_state = np.array([-radius*0.8,0.])
        elif start=='E':
            self.init_state = np.array([radius*0.8,0.])
        if n_forage_spots==1:
            self.goal_states = np.array([[0.75,0.75]])
        elif n_forage_spots==2:
            self.goal_states = np.array([[0.75,0.75], [-0.75,0.75]])
        else:
            self.goal_states = np.random.rand(n_forage_spots,2)*2 - 1
        self.goal_states = self.goal_states*radius/np.sqrt(2)

        self.bear_state = np.array([0.,0.])
        self.bear_radius = bear_radius
        self.bear_penalty = bear_penalty
        self.bear_speed = bear_speed
        self.chase = True
        self.agent_size = agent_size
        self.agent_view_radius = agent_view_radius

        self.render_trace = render_trace
        self.render_mode = render_mode
        self.traces = []

        # If True, show the updated display each time render is called rather
        # than storing the frames and creating an animation at the end
        self.live_display = live_display

        self.lmu_q = 8
        self.lmu = LMU(theta=max_steps//2, q=self.lmu_q, size_in=2)

        self.reward_type = reward_type
        if self.reward_type == 'active':
            self.action_space = spaces.Box(-action_radius, action_radius, (3,), dtype="float64")
            self.penalty = -0.05
        else:
            self.action_space = spaces.Box(-action_radius, action_radius, (2,), dtype="float64")

        self.observation_space = spaces.Box(-radius, radius, (2 + 2*self.lmu_q,), dtype="float64")

        self.bear_sprite = Image.open('gym_water_maze/sprites/black_bear.png')  # Load the bear sprite image
        self.safe_house_sprite = Image.open('gym_water_maze/sprites/BrickHouse.png')  # Load the safe house sprite image
        self.agent_sprite = Image.open('gym_water_maze/sprites/jl_sprite1.png')

        


    def step(self, action):
        if self.reward_type == 'active':
            lick = action[-1] > 0
            action = action[:-1]
        else:
            lick = True
        self.num_steps += 1
        reward = 0
        new_state = self.state + action
        if np.sqrt(np.sum(new_state**2)) > self.radius:
            #reward += -0.001
            rs = np.roots([np.sum(action**2) , 2*np.sum(self.state*action),
                     np.sum(self.state**2)-self.radius**2])
            if np.all(rs<0):
                new_state=self.state
            else:
                new_state = self.state + rs[np.where(rs>=0)[0][0]]*action
        self.state = new_state
        self.traces.append(self.state)
        self.reward_times += self.dt
        dists = np.sqrt(np.sum( (self.state-self.goal_states)**2, axis=-1 ))
        self.reward_probs = 1 - (1 - self.reward_base_probs)**(self.reward_times + 1)
        if lick and np.any(dists <= self.goal_radius):
            idx = np.arange(self.n_forage_spots) == np.argmin(dists)
            if np.random.rand() < self.reward_probs[idx].item():
                reward  = 1
            else:
                reward = 0
            if self.reward_times[idx]>self.dt:
                self.reward_times[idx] = 0
            else:
                self.reward_times[idx] -= 2*self.dt
        if lick and np.all(dists > self.goal_radius):
            reward = self.penalty
        terminated = False

        bear_vec = self.state-self.bear_state
        bear_dist = np.sqrt(np.sum( bear_vec**2 ))
        if bear_dist <= self.bear_radius:
            reward += self.bear_penalty
        
        self.bear_state += self.bear_speed*bear_vec/bear_dist

        if self.num_steps >= self.max_steps:
            truncated = True
        else:
            truncated = False
        # Additional info
        info = {}

        # new
        if bear_dist < self.agent_view_radius:
            self.lmu.step(self.bear_state, chase=self.chase);
        obs = self.make_obs()
        return obs, reward, terminated, truncated, info

    def make_obs(self):
        # vecs = self.state-self.goal_states
        # dists = np.sqrt(np.sum( vecs**2, axis=-1 ))
        # vecs[np.argmin(dists), 0], vecs[np.argmin(dists), 1]

        # obs = np.array([self.state[0],self.state[1], self.bear_state[0], self.bear_state[1]])
        obs = np.concatenate([self.state.flatten(), self.lmu.state.flatten()]).flatten()
        return obs

    def reset(self, seed=None, **kwargs):
        self.seed = seed
        self.num_steps = 0
        self.state = self.init_state.copy()
        self.reward_times = np.zeros(self.n_forage_spots)
        self.reward_probs = np.zeros(self.n_forage_spots)
        # Clean the list of ax_imgs, the buffer for generating videos
        self.ax_imgs = []
        # Clean the traces of the trajectory
        self.traces = [self.init_state]
        self.bear_state = np.array([0.,0.])
        self.chase = random.choice([True, False])

        dists = np.sqrt(np.sum( (self.state-self.goal_states)**2, axis=-1 ))
        bear_dist = np.sqrt(np.sum( (self.state-self.bear_state)**2 ))
        self.lmu.reset();
        obs = self.make_obs()
        return obs, {}

    def render(self,**kwargs):
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_xlim([-self.radius - 0.1, self.radius + 0.1])
        ax.set_aspect('equal')
        ax.set_axis_off()

        angles = np.linspace(0, 2*np.pi, 100)
        xs = self.radius*np.cos(angles)
        ys = self.radius*np.sin(angles)
        ax.plot(xs,ys, linewidth=2, color='k')
        for i, goal_state in enumerate(self.goal_states):
            goal_circ = plt.Circle(goal_state, self.goal_radius, color=plt.cm.Greens(self.reward_probs[i]),clip_on=True )
            ax.add_patch(goal_circ)

        if self.render_trace:
            trace = np.array(self.traces)
            ax.plot(trace[:,0],trace[:,1], '-', color='lightgrey', linewidth=1)
        ax.plot(self.bear_state[0],self.bear_state[1], 'rx', markersize=10)
        ax.plot(self.state[0],self.state[1], 'b.', markersize=10)
        
        # bear_img = ax.imshow(self.bear_sprite, extent=(self.bear_state[1], self.bear_state[1] + self.bear_radius,
        #                                                self.bear_state[0], self.bear_state[0] + self.bear_radius), origin = 'lower')

        # # Draw the agent
        # # ax.plot(self.agent_pos[1], self.agent_pos[0], 'bo')  # Plot the agent's position as a blue dot
        # agent_img = ax.imshow(self.agent_sprite, extent=(self.state[1], self.state[1] + self.agent_size,
                                                         # self.state[0], self.state[0] + self.agent_size), origin = 'lower')
        
        if self.render_mode=='human':
            return fig
        elif self.render_mode=='rgb_array':
            # Retrieve a view on the renderer buffer
            fig.canvas.draw()
            buf = fig.canvas.tostring_rgb()
            ncols, nrows = fig.canvas.get_width_height()
            image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
            return image



    def _get_video(self, interval=200, gif_path=None):
        if self.live_display:
            # TODO: Find a way to create animations without slowing down the live display
            print('Warning: Generating an Animation when live_display=True not yet supported.')
        anim = animation.ArtistAnimation(self.fig, self.ax_imgs, interval=interval)

        if gif_path is not None:
            anim.save(gif_path, writer='imagemagick', fps=10)
        return anim
