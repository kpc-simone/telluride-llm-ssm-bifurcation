import csv
import json
import os
import time
from glob import glob
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, Union

import gymnasium as gym
import pandas
from gymnasium.core import ActType, ObsType


class Monitor(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    '''
    A monitor wrapper for Gymnasium environments, it is used to know the episode reward, length, time and other data.

    :param env: The environment
    :param filename: the location to save a log file, can be None for no log
    :param allow_early_resets: allows the reset of the environment before it is done
    :param reset_keywords: extra keywords for the reset call,
        if extra parameters are needed at reset
    :param info_keywords: extra information to log, from the information return of env.step()
    :param override_existing: appends to file if ``filename`` exists, otherwise
        override existing files (default)
    '''

    EXT = "monitor.csv"

    def __init__(
        self,
        env: gym.Env,
        filename: Optional[str] = None,
        allow_early_resets: bool = True,
        reset_keywords: Tuple[str, ...] = (),
        info_keywords: Tuple[str, ...] = (),
        override_existing: bool = True,
    ):
        super().__init__(env=env)
        self.t_start = time.time()
        self.results_writer = None
        if filename is not None:
            env_id = env.spec.id if env.spec is not None else None
            self.results_writer = ResultsWriter(
                filename,
                header={"t_start": self.t_start, "env_id": str(env_id)},
                extra_keys=reset_keywords + info_keywords,
                override_existing=override_existing,
            )

        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards: List[float] = []
        self.needs_reset = True
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_times: List[float] = []
        self.total_steps = 0
        # extra info about the current episode, that was passed in during reset()
        self.current_reset_info: Dict[str, Any] = {}

    def reset(self, **kwargs) -> Tuple[ObsType, Dict[str, Any]]:
            '''
            Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

            :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
            :return: the first observation of the environment
            '''
            if not self.allow_early_resets and not self.needs_reset:
                raise RuntimeError(
                    "Tried to reset an environment before done. If you want to allow early resets, "
                    "wrap your env with Monitor(env, path, allow_early_resets=True)"
                )
            self.rewards = []
            self.needs_reset = False
            for key in self.reset_keywords:
                value = kwargs.get(key)
                if value is None:
                    raise ValueError(f"Expected you to pass keyword argument {key} into reset")
                self.current_reset_info[key] = value
            return self.env.reset(**kwargs)


    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
            '''
            Step the environment with the given action

            :param action: the action
            :return: observation, reward, terminated, truncated, information
            '''
            if self.needs_reset:
                raise RuntimeError("Tried to step environment that needs reset")
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.rewards.append(float(reward))
            if terminated or truncated:
                self.needs_reset = True
                ep_rew = sum(self.rewards)
                ep_len = len(self.rewards)
                ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
                for key in self.info_keywords:
                    ep_info[key] = info[key]
                self.episode_returns.append(ep_rew)
                self.episode_lengths.append(ep_len)
                self.episode_times.append(time.time() - self.t_start)
                ep_info.update(self.current_reset_info)
                if self.results_writer:
                    self.results_writer.write_row(ep_info)
                info["episode"] = ep_info
            self.total_steps += 1
            return observation, reward, terminated, truncated, info


    def close(self) -> None:
            """
            Closes the environment
            """
            super().close()
            if self.results_writer is not None:
                self.results_writer.close()

    def get_total_steps(self) -> int:
            '''
            Returns the total number of timesteps

            :return:
            '''
            return self.total_steps


    def get_episode_rewards(self) -> List[float]:
            '''
            Returns the rewards of all the episodes

            :return:
            '''
            return self.episode_returns


    def get_episode_lengths(self) -> List[int]:
            """
            Returns the number of timesteps of all the episodes

            :return:
            """
            return self.episode_lengths


    def get_episode_times(self) -> List[float]:
            '''
            Returns the runtime in seconds of all the episodes

            :return:
            '''
            return self.episode_times
            
    def get_rewards(self) -> List[float]:
            
            return self.rewards
            
    def get_timesteps(self) -> List[float]:
            
            return [ t for t in range(self.total_steps) ]