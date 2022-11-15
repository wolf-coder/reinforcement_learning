#!/usr/bin/env python3
"""
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
function that has the trained agent play an episode:
  - env is the FrozenLakeEnv instance
  - Q is a numpy.ndarray containing the Q-table
  - max_steps is the maximum number of steps in the episode
  - Each state of the board should be displayed via the console
  - You should always exploit the Q-table
  - Returns: the total rewards for the episode
    """
    total_rewards = 0
    state = env.reset()[0]
    done = False
    print(env.render())
    for step in range(max_steps):

        action = np.argmax(Q[state, :])
        new_state, reward, done, info, prob = env.step(action)

        state = new_state
        total_rewards += reward

        # rendering
        print(env.render())
        if done:
            break

    return total_rewards
