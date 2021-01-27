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


def pull_vegas_action(history, explore_first = False):
    # all array has a size of (machine_n, )
    self_pull = np.nansum(history[:, SELF_PULL_INDEX, :], axis = -1)
    opp_pull = np.nansum(history[:, OPP_PULL_INDEX, :], axis = -1)
    self_win = np.nansum(history[:, SELF_REWARD_INDEX, :], axis = -1)
    self_loss = self_pull - self_win
    total_pull = self_pull + opp_pull 
    # compute pull vegas score
    scores = (1. + self_win - self_loss + opp_pull - (opp_pull > 0)*1.5) / (1. + total_pull)

    # make decision
    if not explore_first:
        my_pull = np.argmax(scores)
    else:
        best_score = scores.max()
        best_pulls = (scores == best_score).nonzero()[0]
        total_pulls = np.nansum(history[:, [SELF_PULL_INDEX, OPP_PULL_INDEX], :], axis = (1, 2))
        unvisited_pulls = (total_pulls == 0.).nonzero()[0]
        assert len(total_pulls) == MACHINE_N
        intersect_pulls = np.intersect1d(best_pulls, unvisited_pulls)
        if len(intersect_pulls) != 0:
            candidate_pulls = list(intersect_pulls)
        else:
            candidate_pulls = list(best_pulls)
        my_pull = random.sample(candidate_pulls, 1)[0]

    return int(my_pull)


def pull_with_consec_k(history, t, k, is_opp = True):
    target_idx = OPP_PULL_INDEX if is_opp else SELF_PULL_INDEX
    start_t = t - k + 1
    end_t = t + 1
    # (machine_n, )
    consec_sum_pulls = np.nansum(history[:, target_idx, start_t:end_t], axis = -1)
    target_pulls = (consec_sum_pulls == k).nonzero()[0]
    target_pull = None if len(target_pulls) == 0. else int(list(target_pulls)[0])
    return target_pull


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
        opp_consec_pull = pull_with_consec_k(history, t, k = 2, is_opp = True)
        if opp_consec_pull is not None:
            my_pull = opp_consec_pull
        else:
            my_pull = pull_vegas_action(history, explore_first = True)
    
    return my_pull
    