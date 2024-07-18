import gymnasium as gym
import nengo
from functools import reduce
import operator
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import sys,os

sys.path.insert(0,'gym-tellurun')
import gym_tellurun

def image_svg(arr):
    """
    Given an array, return an svg image. Only supported in Python 3.
    """

    png = Image.fromarray(arr)
    buffer = BytesIO()
    png.save(buffer, format="PNG")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    return '''
        <svg width="100%%" height="100%%" viewbox="0 0 100 100">
        <image width="100%%" height="100%%"
               xlink:href="data:image/png;base64,%s"
               style="image-rendering: pixelated;"/>
        </svg>''' % (''.join(img_str))


class GymEnv(object):

    def __init__(self,
                 env_name='CartPole-v1',
                 env=None,
                 scale_input=True,
                 reset_signal=False,
                 reset_when_done=False,
                 return_reward=True,
                 return_done=False,
                 render=True,
                 nengo_steps_per_update=100,
                ):
        """
        env_name (str): name of gym environment to create
        scale_input (bool): if true, scale the input from -1:+1 to the action space
        reset_signal (bool): if true, one of the inputs will be a reset signal
        reset_when_done (bool): if true, automatically reset the environment on the done signal
        return_reward (bool): if true, return the reward as an output
        return_done (bool): if true, return the done signal as an output
        render (bool): if true, call render after each step
        nengo_steps_per_update (int): number of Nengo simulation steps per gym step
        """

        self.env_name = env_name

        self.scale_input = scale_input
        self.reset_signal = reset_signal
        self.reset_when_done = reset_when_done
        self.return_reward = return_reward
        self.return_done = return_done
        self.render = render
        if render:
            render_mode="rgb_array"
            self._nengo_html_ = ""  # initialize HTML property for plotting in GUI
        else:
            render_mode=None

        self.nengo_steps_per_update = nengo_steps_per_update
        self.nengo_steps = 0

        self.env = gym.make(self.env_name, render_mode=render_mode)
            
        # Dimensionality of actions
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.act_dim = 1
            self.action_type = 'Discrete'
        elif isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
            self.act_dim = self.env.action_space.nvec.size
            self.action_type = 'MultiDiscrete'
        elif isinstance(self.env.action_space, gym.spaces.Box):
            self.act_dim = self.env.action_space.shape[0]
            self.action_type = 'Box'
        else:
            raise NotImplementedError("Action space dimensionality for {0} not implemented".format(type(self.env.action_space)))

        # Dimensionality of observations
        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            self.obs_dim = 1
        elif isinstance(self.env.observation_space, gym.spaces.MultiDiscrete):
            self.obs_dim = self.env.observation_space.nvec.size
        elif isinstance(self.env.observation_space, gym.spaces.Box):
            #self.obs_dim = self.env.observation_space.shape[0]
            self.obs_dim = reduce(operator.mul, self.env.observation_space.shape)
        else:
            raise NotImplementedError("Observation space dimensionality for {0} not implemented".format(type(self.env.observation_space)))

        self.size_in = self.act_dim
        self.size_out = self.obs_dim

        self.obs, info = self.env.reset()
        self.reward = 0 
        self.done = 0
        outputs = [self.obs.flatten()]
        if self.return_reward:
            outputs.append([self.reward])
            self.size_out += 1
        if self.return_done:
            outputs.append([self.done])
            self.size_out += 1
        if self.reset_signal:
            self.size_in += 1

        self.output = np.concatenate(outputs)

    def __call__(self, t, x):

        self.nengo_steps += 1
        if self.nengo_steps >= self.nengo_steps_per_update:
            self.nengo_steps = 0
            
            action = x[:self.act_dim]
            
            if self.action_type == 'Discrete':
                if self.scale_input:
                    action = int((action + 1) * self.env.action_space.n / 2.)
                else:
                    action = int(action)
                action = np.clip(action, 0, self.env.action_space.n - 1)
            elif self.action_type == 'MultiDiscrete':
                pass # TODO
            elif self.action_type == 'Box':
                if self.scale_input:
                    # TODO: handle infinite limits on action space nicely
                    action = ((action + 1) / 2.)*\
                            (self.env.action_space.high - self.env.action_space.low) +\
                            self.env.action_space.low
                action = np.clip(
                    action,
                    self.env.action_space.low,
                    self.env.action_space.high
                )
            else:
                if self.scale_input:
                    action = int((action + 1) * self.env.action_space.n / 2.)
                else:
                    action = int(action)
                action = np.clip(action, 0, self.env.action_space.n - 1)
                print(action)

            if self.reset_signal:
                # reset when the signal is nonzero
                reset = int(x[self.act_dim])
            else:
                reset = 0

            # NOTE: if done came from the environment, it will be reset on the next cycle through
            #       if the reset signal is set, it will reset on the current cycle
            if (self.reset_when_done and self.done) or (self.reset_signal and reset):
                self.obs, info = self.env.reset()
                self.reward = 0
                self.done = 0
            else:
                self.obs, self.reward, self.done, truncated, info = self.env.step(action)

            if self.render:
                render_data = self.env.render()
                self._nengo_html_ = image_svg(render_data)

            outputs = [self.obs.flatten()]
            if self.return_reward:
                outputs.append([self.reward])
            if self.return_done:
                outputs.append([self.done])

            self.output = np.concatenate(outputs)

        return self.output

    def close(self):

        self.env.close()