import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
from matplotlib.backends.backend_agg import FigureCanvasAgg

import gymnasium as gym
from gymnasium import spaces


# adapted from https://github.com/rpinsler/gym-maze/blob/master/gym_maze/envs/maze.py
class RelativeWaterMazeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self,
                 radius = 6,
                 goal_radius = 1,
                 action_radius = 1,
                 n_spots = 2,
                 dtheta_spots = 0.5,
                 goal_idx = 0,
                 dt= 0.001,
                 max_steps = 100,
                 render_mode = 'human',
                 reward_type= 'sparse', # sparse or active
                 live_display=False,
                 render_trace=False):
        """Initialize the maze. DType: list"""
        self.max_steps = max_steps
        self.radius = radius
        self.n_spots = n_spots
        self.goal_radius = goal_radius
        self.dt = dt
        self.init_state = np.array([0.,0.])

        self.dtheta_spots = dtheta_spots
        self.goal_states = np.zeros((n_spots,2))
        self.goal_states[0,:] = np.random.rand(2)*2 - 1
        self.rotation_matrix = np.array([[np.cos(self.dtheta_spots), -np.sin(self.dtheta_spots)],[np.sin(self.dtheta_spots), np.cos(self.dtheta_spots)]])
        for i in range(1,n_spots):
            self.goal_states[i,:] = self.rotation_matrix @ self.goal_states[i-1,:]
        self.goal_states = self.goal_states*radius/np.sqrt(2)
        self.goal_idx = goal_idx
        
        self.render_trace = render_trace
        self.render_mode = render_mode
        self.traces = []

        # If True, show the updated display each time render is called rather
        # than storing the frames and creating an animation at the end
        self.live_display = live_display

        self.reward_type = reward_type
        if self.reward_type == 'active':
            self.action_space = spaces.Box(-action_radius, action_radius, (3,), dtype="float64")
            self.penalty = -0.05
        else:
            self.action_space = spaces.Box(-action_radius, action_radius, (2,), dtype="float64")
        self.observation_space = spaces.Box(-radius, radius, (3,), dtype="float64")


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
        all_dist = np.sqrt(np.sum( (self.state-self.goal_states)**2, axis=-1 ))
        at_goal = np.any(all_dist<self.goal_radius)
        dist = np.sqrt(np.sum( (self.state-self.goal_states[self.goal_idx])**2, axis=-1 ))
        terminated = False
        if lick and (dist <= self.goal_radius):
            reward = 1 - 0.9*(self.num_steps/self.max_steps)
            terminated = True
        if lick and (dist > self.goal_radius):
            reward = self.penalty
        if self.num_steps >= self.max_steps:
            truncated = True
        else:
            truncated = False
        # Additional info
        info = {}
        return np.concatenate([self.state,[at_goal]]), reward, terminated, truncated, info


    def reset(self, seed=None, **kwargs):
        self.seed = seed
        self.num_steps = 0
        self.state = self.init_state.copy()
        self.goal_states = np.zeros((self.n_spots,2))
        self.goal_states[0,:] = np.random.rand(2)*2 - 1
        self.rotation_matrix = np.array([[np.cos(self.dtheta_spots), -np.sin(self.dtheta_spots)],[np.sin(self.dtheta_spots), np.cos(self.dtheta_spots)]])
        for i in range(1,self.n_spots):
            self.goal_states[i,:] = self.rotation_matrix @ self.goal_states[i-1,:]
        self.goal_states = self.goal_states*self.radius/np.sqrt(2)
        # Clean the list of ax_imgs, the buffer for generating videos
        self.ax_imgs = []
        # Clean the traces of the trajectory
        self.traces = [self.init_state]
        all_dist = np.sqrt(np.sum((self.state - self.goal_states) ** 2, axis=-1))
        at_goal = np.any(all_dist < self.goal_radius)
        return np.concatenate([self.state,[at_goal]]), {}

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
            if i==self.goal_idx:
                goal_circ = plt.Circle(goal_state, self.goal_radius, color=plt.cm.Blues(0.5),clip_on=True )
                ax.add_patch(goal_circ)
            else:
                goal_circ = plt.Circle(goal_state, self.goal_radius, color=plt.cm.Greens(1),clip_on=True )
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
