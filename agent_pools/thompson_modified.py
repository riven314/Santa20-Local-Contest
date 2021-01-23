"""
modified thompson sampling from Vic
"""
import time
import numpy as np


bandit = None
total_reward = 0
n_machines=100
n_success_n_pull=[[] for _ in range(n_machines)]
n_pull=np.array([0 for _ in range(n_machines)])
prob_arrays = [[0.01]*100 for _ in range(n_machines)]


def cond_prob(theta: float, obs_list: list): #p(theta|obs_list)
    func_value = 1
    index = 0
    while index < len(obs_list):
        obs = obs_list[index]
        func_value = func_value*((theta*0.97**obs[1])*(-1)**(1-obs[0])+1-obs[0])
        index += 1
    return func_value


def prior_conj(theta:float,obs_list:list): #integrate p(theta|obs_list) from 0 to 1
    nom = cond_prob(theta,obs_list)
    denom = 1. 
    #quad(cond_prob,0,1,args=obs_list,epsabs=0.02)[0]
    return nom / denom


def prob_array(obs_list:list,n_ele:int): #integrate p(theta|obs_list) from 0 to 1
    dis_supp = np.linspace(0,1, n_ele)
    dis = np.asarray([prior_conj(theta, obs_list) for theta in dis_supp])
    prob = dis/dis.sum()
    return prob


def sample(prob):
    sample = np.random.choice(
        a = np.linspace(0,1, len(prob)),
        size = None, replace = True, p = prob
    )
    return sample


def plot_check(obs_list,finess):
    import matplotlib.pyplot as plt
    t=np.arange(0.,1.,finess)
    plt.plot(t,list(map(lambda theta:prior_conj(theta,obs_list),t)))    
    result=0
    finess=0.001
    for i in np.arange(0.,1.,finess):
        result+=finess*prior_conj(i,obs_list)    
    #print(result)
    return None


def agent(observation, configuration):
    global total_reward, bandit, n_machines, n_pull,n_success_n_pull,last_reward,prob_arrays
    
    if observation.step == 0:
        
        bandit = None
        total_reward = 0
        n_machines = 100
        n_success_n_pull = [[] for _ in range(n_machines)]
        n_pull = np.array([0 for _ in range(n_machines)])
        prob_arrays = [[0.01]*100 for _ in range(n_machines)]


    last_reward = observation['reward'] - total_reward
    total_reward = observation['reward']

    if len(observation['lastActions']) == 2:
        # Update number of pulls for both machines
        m_index = observation['lastActions'][observation['agentIndex']]
        opp_index = observation['lastActions'][(observation['agentIndex'] + 1) % 2]
        n_pull[m_index] += 1
        n_pull[opp_index] += 1
        n_success_n_pull[m_index].append((last_reward, n_pull[m_index]))
  
        # Update the distribution for the machine pulled last turn
        prob_arrays[m_index] = prob_array(
            obs_list = n_success_n_pull[m_index], n_ele = 500)

    discounted_samples = np.asarray([sample(prob_arrays[_])*0.97**(n_pull[_]) for _ in range(n_machines)])
    bandit = int(np.argmax(discounted_samples))
    
    #print("Step:"+str(observation.step))
    #print("Bandit:"+str(bandit))
    return bandit