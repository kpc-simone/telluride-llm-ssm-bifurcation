import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
from matplotlib.backends.backend_agg import FigureCanvasAgg

import gymnasium as gym
from gymnasium import spaces


# adapted from https://github.com/rpinsler/gym-maze/blob/master/gym_maze/envs/maze.py
class ForageWaterMazeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self,
                 radius = 6,
                 goal_radius = 1,
                 action_radius = 1,
                 n_forage_spots = 2,
                 dt= 0.001,
                 start = 'S', #S,E,W,N
                 max_steps = 100,
                 render_mode = 'human',
                 live_display=False,
                 render_trace=False):
        """Initialize the maze. DType: list"""
        self.max_steps = max_steps
        self.radius = radius
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
            self.init_state = np.array([0,-radius*0.8])
        elif start=='N':
            self.init_state = np.array([0,radius*0.8])
        elif start=='W':
            self.init_state = np.array([-radius*0.8,0])
        elif start=='E':
            self.init_state = np.array([radius*0.8,0])
        if n_forage_spots==1:
            self.goal_states = np.array([[0.75,0.75]])
        elif n_forage_spots==2:
            self.goal_states = np.array([[0.75,0.75], [-0.75,0.75]])
        else:
            self.goal_states = np.random.rand(n_forage_spots,2)*2 - 1
        self.goal_states = self.goal_states*radius/np.sqrt(2)

        self.render_trace = render_trace
        self.render_mode = render_mode
        self.traces = []

        # If True, show the updated display each time render is called rather
        # than storing the frames and creating an animation at the end
        self.live_display = live_display

        self.action_space = spaces.Box(-action_radius, action_radius, (2,), dtype="float64")
        self.observation_space = spaces.Box(-radius, radius, (2,), dtype="float64")


    def step(self, action):
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
        if np.any(dists <= self.goal_radius):
            idx = np.arange(self.n_forage_spots) == np.argmin(dists)
            if np.random.rand() < self.reward_probs[idx].item():
                reward  = 1
            else:
                reward = 0
            if self.reward_times[idx]>self.dt:
                self.reward_times[idx] = 0
            else:
                self.reward_times[idx] -= 2*self.dt

        terminated = False

        if self.num_steps >= self.max_steps:
            truncated = True
        else:
            truncated = False
        # Additional info
        info = {}
        return self.state, reward, terminated, truncated, info


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
        return self.state, {}

    def render(self,**kwargs):
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_xlim([-self.radius - 0.1, self.radius + 0.1])
        ax.set_aspect('equal')

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
        ax.plot(self.state[0],self.state[1], 'rx', markersize=10)
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
