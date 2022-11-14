#!/usr/bin/env python3
"""
Using epsilon-greedy to determine the next action.
"""
import random
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
Function that uses epsilon-greedy to determine the next action:
  - Q is a numpy.ndarray containing the q-table
  - state is the current state
  - epsilon is the epsilon to use for the calculation
  - You should sample p with numpy.random.uniformn to determine if your
algorithm should explore or exploit
  - If exploring, you should pick the next action with numpy.random.randint
from all possible actions
  - Returns: the next action index
    """
    # First we randomize a number
    exp_exp_tradeoff = np.random.uniform(0, 1)

    # If this number > greater than epsilon --> exploitation (taking
    # the biggest Q value for this state)
    if exp_exp_tradeoff > epsilon:
        action = np.argmax(Q[state, :])

    # Else doing a random choice --> exploration
    else:
        action = np.random.randint(Q.shape[1])

    return action
