"""
modified from orig pull vegas:
https://www.kaggle.com/sirishks/pull-vegas-slot-machines
"""
from random import sample
import numpy as np
import pandas as pd
import random, os, datetime

total_reward = 0
bandit_dict = {}
bandit_visited = [False] * 100
bandit_visited[0] = True


def set_seed(my_seed = 42):
    os.environ['PYTHONHASHSEED'] = str(my_seed)
    random.seed(my_seed)
    np.random.seed(my_seed)


def get_next_bandit():
    bandit_expect = np.array([0.] * 100)
    best_bandit_expected = 0
    for bnd in bandit_dict:
        my_diff = bandit_dict[bnd]['win'] - bandit_dict[bnd]['loss']
        opp_diff = bandit_dict[bnd]['opp'] - (bandit_dict[bnd]['opp']>0)*1.5
        total_pull = (bandit_dict[bnd]['win'] + bandit_dict[bnd]['loss'] + bandit_dict[bnd]['opp'])
        expect = (my_diff + opp_diff) / total_pull
        bandit_expect[bnd] = expect
        if expect > best_bandit_expected:
            best_bandit_expected = expect
    
    unvisited_bandits = (bandit_visited == False).nonzero()[0]
    best_bandits = (bandit_expect == best_bandit_expected).nonzero()[0]
    intersect_bandits = np.intersect1d(unvisited_bandits, best_bandits)
    if len(intersect_bandits) != 0:
        # randomly select bandit that are unvisited & best expect score
        candidate_bandits = list(intersect_bandits)
    else:
        # randomly select bandits with best score
        candidate_bandits = list(best_bandits)

    my_bandit = sample(candidate_bandits, 1)[0]
    return my_bandit


def multi_armed_probabilities(observation, configuration):
    global total_reward, bandit_dict, bandit_visited

    my_pull = random.randrange(configuration['banditCount'])
    if 0 == observation['step']:
        set_seed()
        total_reward = 0
        bandit_dict = {}
        for i in range(configuration['banditCount']):
            bandit_dict[i] = {'win': 1, 'loss': 0, 'opp': 0}
    else:
        last_reward = observation['reward'] - total_reward
        total_reward = observation['reward']
        
        my_idx = observation['agentIndex']
        my_last_pull = observation['lastActions'][my_idx]
        bandit_visited[my_last_pull] = True

        if last_reward > 0:
            bandit_dict[observation['lastActions'][my_idx]]['win'] = bandit_dict[observation['lastActions'][my_idx]]['win'] + 1
        else:
            bandit_dict[observation['lastActions'][my_idx]]['loss'] = bandit_dict[observation['lastActions'][my_idx]]['loss'] + 1
        bandit_dict[observation['lastActions'][1-my_idx]]['opp'] = bandit_dict[observation['lastActions'][1-my_idx]]['opp'] + 1
        # put your action into queue
        my_pull = get_next_bandit()
    return my_pull