import os
import random
import numpy as np


MACHINE_N = 100
TIMESTEPS = 2000
SELF_PULL_INDEX = 0
SELF_REWARD_INDEX = 1
OPP_PULL_INDEX = 2

history = np.zeros((MACHINE_N, 3, TIMESTEPS))
history[:, :, :] = np.nan


def set_seed(my_seed = 42):
    os.environ['PYTHONHASHSEED'] = str(my_seed)
    random.seed(my_seed)
    np.random.seed(my_seed)


def pull_vegas_action(history):
    # all array has a size of (machine_n, )
    self_pull = np.nansum(history[:, SELF_PULL_INDEX, :], axis = -1)
    opp_pull = np.nansum(history[:, OPP_PULL_INDEX, :], axis = -1)
    self_win = np.nansum(history[:, SELF_REWARD_INDEX, :], axis = -1)
    self_loss = self_pull - self_win
    total_pull = self_pull + opp_pull 
    # compute pull vegas score
    scores = (1. + self_win - self_loss + opp_pull - (opp_pull > 0)*1.5) / (1. + total_pull)
    # make decision
    #ranked_bandits = np.argsort(scores)[::-1]
    #my_pull = ranked_bandits[0]
    my_pull = np.argmax(scores)
    return int(my_pull)


def _update_history(history, observation, t):
    # parse last timestep stats
    my_idx = observation['agentIndex']
    opp_idx = 1 - my_idx
    last_bandit = observation['lastActions'][my_idx]
    opp_last_bandit = observation['lastActions'][opp_idx]
    outdated_last_reward = np.nansum(history[:, SELF_REWARD_INDEX, :])
    last_reward = observation['reward'] - outdated_last_reward
    assert (last_reward == 0) or (last_reward == 1.)

    # update history
    history[:, :, t] = 0.
    history[last_bandit, SELF_PULL_INDEX, t] = 1.
    history[last_bandit, SELF_REWARD_INDEX, t] = last_reward
    history[opp_last_bandit, OPP_PULL_INDEX, t] = 1.
    return history


def strategy(observation, configuration):
    global history
    t = observation['step']
    
    if t == 0:
        set_seed()
        my_pull = random.randrange(configuration['banditCount'])    
        history[:, :, t] = 0.
    else:
        history = _update_history(history, observation, t)
        my_pull = pull_vegas_action(history)
    
    return my_pull
    