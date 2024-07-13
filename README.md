# Neuomodulation of SSMs for RL

For the code in the modulating-theta-nengo, current-injection-nengo, and lmu-rl-nengo folders, you will need nengo:
<pre><code>pip install nengo nengogui matplotlib scipy </code></pre>
If you would like to speed up the simulation time install nengo-ocl and just replace nengo.Simulator with nengo_ocl.Simulator

For the code in gym-water-maze and rl-agent, install gymnasium and a version of torch based on your version of CUDA. For CPU:
<pre><code>pip install gymnasium</code></pre>
<pre><code>pip install torch torchvision torchaudio</code></pre>

For the gym-water-maze package:
<pre><code>pip install stable-baselines3</code></pre>
<pre><code>cd ./gym-water-maze</code></pre>
<pre><code>pip install .</code></pre>
stable-baselines3 is just used for testing the gym environment.


### To do
- Add a new environment to gym-water-maze or modify the foraging one to include a moving negative spot and add graphics
- Add code for rl-agent: this will be adapted from successsor-features-A2C but with unnecessary code and options removed. May also see if extending rl-zoo if possible
- Modify the rl-agent code to include an LMU memory on the observations and add a connection for current/bias injection
- Modify the rl-agent code to include conceptors
- Combine the modulating theta + current injection and nengo rl folders and integrate modulation into the simple rl value agents
- Add actor to the nengo rl code 
