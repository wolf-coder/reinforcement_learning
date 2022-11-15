#!/usr/bin/env python3
"""
Loading the pre-made FrozenLakeEnv evnironment from OpenAI’s gym
"""
import numpy as np
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
Function that loads the pre-made FrozenLakeEnv evnironment from OpenAI’s gym:
  + desc is either None or a list of lists containing a custom
description of the map to load for the environment
  + map_name is either None or a string containing the pre-made map to load
  + Note: If both desc and map_name are None, the environment
will load a randomly generated 8x8 map
  + is_slippery is a boolean to determine if the ice is slippery
  + Returns: the environment
    """
    if desc is None and map_name is None:
        desc = generate_random_map(size=8)

    env = gym.make('FrozenLake-v1',
                   desc=desc,
                   map_name=map_name,
                   is_slippery=is_slippery,
                   render_mode="ansi")

    return env
