import nengo
import matplotlib
import sys,os
import numpy as np
#print(matplotlib.__version__)

#import subprocess
#def install(package):
#    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

#install('matplotlib==3.7.0')
#import matplotlib
#print(matplotlib.__version__)

from nengo_gym import GymEnv
import gymnasium

model = nengo.Network(seed=13)

inputs = [2,1,0,3]

with model:

    # dt of CartPole is 0.02
    # dt of Nengo is 0.001
    env = GymEnv(
        env_name='TelluRUN-v0',
        reset_signal=False,
        reset_when_done=True,
        return_reward=True,
        return_done=False,
        render=True,
        nengo_steps_per_update=1,
    )

    TelluRUN = nengo.Node(
        env,
        size_in=env.size_in,
        size_out=env.size_out
    )

#    action = nengo.Node( lambda t: env.env.action_space.sample() )
#    action = nengo.Node( nengo.processes.PresentInput(inputs, presentation_time=0.1) )
    action = nengo.Node([1])

    nengo.Connection(action, TelluRUN)

def on_close(sim):
    env.close()


if __name__ == '__main__':

    sim = nengo.Simulator(model)
    sim.run(10)