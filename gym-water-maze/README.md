## Water maze gym environment 

Includes three gymnasium environments: WaterMaze, ForageWaterMaze, RelativeWaterMaze. All are 2D circular environments with goal locaiton(s). 

State is the agent's (x,y) position in the environment -- a 2D Box space.

Action is either (Δx, Δy) or (Δx, Δy, lick) (if reward_type is active).



### WaterMaze
Single goal location. Rewards can be set to be sparse (receives reward of 1 - (steps taken)/(max steps) when goal location is reached), dense (reward based on distance to goal), active (receives reward of 1 - (steps taken)/(max steps) when goal location is reached and lick is > 0 and a negative reward of 0.05 if lick>0 when not at goal).

### ForageWaterMazeEnv
Many goal locations (default is two). Rewards are sparse and probabilistic. Different goal locations have a different 'base' probability of reward and the more one location is visited, the lower the reward probability becomes while the probability for other locations increases. Reward type can also be set to 'active' in which case lick>0 is also requried for a chance at reward.

### RelativeWaterMazeEnv
Many 'goal' locations (default is two) that change every reset. The 'goal' location most counterclockwise is the true goal. State is (x,y, is_goal) where is_goal is either positive (the agent is on top of a 'goal' location) or negative. This task requries memory. 
