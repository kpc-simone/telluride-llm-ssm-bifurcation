from gymnasium.envs.registration import register
from gym_water_maze.watermazeenv import WaterMazeEnv
from gym_water_maze.foragewatermazeenv import ForageWaterMazeEnv
from gym_water_maze.relativewatermazeenv import RelativeWaterMazeEnv

register(
    id="WaterMaze-v0",
    entry_point="gym_water_maze:WaterMazeEnv"
)

register(
    id="ForageWaterMaze-v0",
    entry_point="gym_water_maze:ForageWaterMazeEnv"
)

register(
    id="RelativeWaterMaze-v0",
    entry_point="gym_water_maze:RelativeWaterMazeEnv"
)

