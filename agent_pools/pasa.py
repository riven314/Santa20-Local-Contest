"""
source:
https://www.kaggle.com/demetrypascal/pasa-select-best
"""
import random
import numpy as np


EXPLORE_STEPS = 5 # count of repeats of random selection before start of main algorithm
PROB = 0.9 # prop for usage algorithm (random otherwise)
ROUNDS = 2000

c_arr = np.empty(ROUNDS) # array of coefs 1, 0.97, 0.97^2, ...
c_arr[0] = 1
for i in range(1, c_arr.size):
    c_arr[i] = c_arr[i-1]*0.97

x_arr = np.linspace(0, 1, 101) # net of predicted thresholds
BANDITS = 100 # count of bandits
bandit_counts = np.zeros(BANDITS, dtype = np.int16) # choices count for each bandit
probs = np.ones((BANDITS, x_arr.size)) # matrix bandit*threshold probs
start_bandits = np.random.choice(np.arange(BANDITS), BANDITS*EXPLORE_STEPS, replace = True) # just start random sequence of bandits selection before start of main algorithm


def update_counts(my_action, opponent_action, my_reward):
    global bandit_counts, probs
    
    if my_reward == 1:
        probs[my_action, :] *= x_arr * c_arr[bandit_counts[my_action]]
    else:
        probs[my_action, :] *= 1 - x_arr * c_arr[bandit_counts[my_action]]
    
    bandit_counts[my_action] += 1
    bandit_counts[opponent_action] += 1


def get_best_action():
    #inds = np.unravel_index(probs.argmax(), probs.shape)
    #return inds[0] # select best bandit
    #likeh = np.array([np.argmax(probs[i, :]) for i in range(BANDITS)])
    #likeh = np.array([x_arr[ind]*c_arr[b]*probs[bandit, ind]/probs[bandit, :].sum() for bandit, (ind, b) in enumerate(zip(likeh, bandit_counts))])

    likeh = np.array([np.random.choice(x_arr, 1, p = probs[bandit, :]/probs[bandit, :].sum())[0] * c_arr[b] for bandit, b in enumerate(bandit_counts)])
    return np.random.choice(BANDITS, 1, p = likeh/likeh.sum()) if random.random() < PROB else random.randrange(BANDITS)

last_reward = 0


def pasa_agent(observation, configuration):
    
    global BANDITS, start_bandits, bandit_counts, probs, last_reward
    
    if observation.step == 0:
        BANDITS = configuration["banditCount"]
        #print(f"there are {BANDITS} bandits")
        start_bandits = np.random.choice(np.arange(BANDITS, dtype = np.int16), BANDITS*EXPLORE_STEPS, replace = True)
        bandit_counts = np.zeros(BANDITS, dtype = np.int16)
        probs = np.ones((BANDITS, x_arr.size))
        my_last_action = start_bandits[0]
    
    elif observation.step < start_bandits.size:
        update_counts(int(observation.lastActions[0]), int(observation.lastActions[1]), observation['reward'] - last_reward)
        my_last_action = start_bandits[observation.step]
    
    else:
        update_counts(int(observation.lastActions[0]), int(observation.lastActions[1]), observation['reward'] - last_reward)
        my_last_action = get_best_action()
    
    last_reward = observation['reward'] 
    return int(my_last_action)