import gymnasium
from gymnasium.envs.registration import register
from gym_tellurun.envs.ChasingBearEnv_xyrep import ChasingBearEnv

gymnasium.envs.register(
     id = 'TelluRUN-v0',
     entry_point = 'gym_tellurun:ChasingBearEnv',
     max_episode_steps = 200,
)