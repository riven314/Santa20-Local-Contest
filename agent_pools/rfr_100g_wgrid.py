"""Greedy agent that chooses machine based on maximum expected payout

Uses a trained decision tree model to consider the other player's movements
in the expected payout.

See my other kernel for methodology for generating training data:
https://www.kaggle.com/lebroschar/generate-training-data

"""
import os
import joblib
import pickle
import random
import base64
from shutil import copyfile

import numpy as np
import pandas as pd
import sklearn.tree as skt
import tarfile
from datetime import datetime

# model path
MODEL_FILENAME = '100g_wtune_rfr.joblib'
try:
    os.listdir("/kaggle_simulations/agent/")
    model_dir = "/kaggle_simulations/agent/"
except:
    model_dir = os.getcwd()
file_name = [fn for fn in os.listdir(model_dir) if fn == MODEL_FILENAME][0]
MODEL_PATH = os.path.join(model_dir, file_name)

# Parameters
FUDGE_FACTOR = 0.99
VERBOSE = False
DATA_FILE = '/kaggle/input/sample-training-data/training_data_201223.parquet'
TRAIN_FEATS = ['round_num', 'n_pulls_self', 'n_success_self', 'n_pulls_opp']
TARGET_COL = 'payout'


def pickle_model(model): #pickle the trained model and output the name of the saved text file
    model_bytes = pickle.dumps(model)
    serialized_string = base64.b64encode(model_bytes)
    txt_name="pickled_model.txt"
    with open(txt_name,"wb+") as pick:
        pick.write(serialized_string)


def make_tarfile(output_filename, file_names):  #save files into tar.gz. files, replaced by !tar
    with tarfile.open(output_filename, "x:gz") as tar:
        for file_name in file_names:
            tar.add(file_name)

        
def make_model():
    """Builds a decision tree model based on stored traininged data"""
    data = pd.read_parquet(DATA_FILE)
    model = skt.DecisionTreeRegressor(min_samples_leaf=40)
    model.fit(data[TRAIN_FEATS], data[TARGET_COL])
    return model


def load_model():
    print(f'model_path: {MODEL_PATH}')
    model = joblib.load(MODEL_PATH)
    return model


class GreedyStrategy:
    """Implements strategy to maximize expected value

    - Tracks estimated likelihood of payout ratio for each machine
    - Tracks number of pulls on each machine
    - Chooses machine based on maximum expected value
    
    
    """
    def __init__(self, name, agent_num, n_machines):
        """Initialize and train decision tree model

        Args:
           name (str):   Name for the agent
           agent_num (int):   Assigned player number
           n_machines (int):   number of machines in the game
        
        """
        # Record inputs
        self.name = name
        self.agent_num = agent_num
        self.n_machines = n_machines
        
        # Initialize distributions for all machines
        self.n_pulls_self = np.array([0 for _ in range(n_machines)])
        self.n_success_self = np.array([0. for _ in range(n_machines)])
        self.n_pulls_opp = np.array([0 for _ in range(n_machines)])

        # Track other players moves
        self.opp_moves = []
        
        # Track winnings
        self.last_reward_count = 0

        # Create model to predict expected reward
        self.model = load_model()
        
        # Predict expected reward
        features = np.zeros((self.n_machines, 4))
        features[:, 0] = len(self.opp_moves)
        features[:, 1] = self.n_pulls_self
        features[:, 2] = self.n_success_self
        features[:, 3] = self.n_pulls_opp
        self.predicts = self.model.predict(features)
        

    def __call__(self):
        """Choose machine based on maximum expected payout

        Returns:
           <result> (int):  index of machine to pull
        
        """
        # Otherwise, use best available
        est_return = self.predicts
        max_return = np.max(est_return)
        result = np.random.choice(np.where(
            est_return >= FUDGE_FACTOR * max_return)[0])
        
        if VERBOSE:
            print('  - Chose machine %i with expected return of %3.2f' % (
                int(result), est_return[result]))

        return int(result)
    
        
    def updateDist(self, curr_total_reward, last_m_indices):
        """Updates estimated distribution of payouts"""
        # Compute last reward
        last_reward = curr_total_reward - self.last_reward_count
        self.last_reward_count = curr_total_reward
        if VERBOSE:
            print('Last reward: %i' % last_reward)

        if len(last_m_indices) == 2:
            # Update number of pulls for both machines
            m_index = last_m_indices[self.agent_num]
            opp_index = last_m_indices[(self.agent_num + 1) % 2]
            self.n_pulls_self[m_index] += 1
            self.n_pulls_opp[opp_index] += 1

            # Update number of successes
            self.n_success_self[m_index] += last_reward
            
            # Update opponent activity
            self.opp_moves.append(opp_index)

            # Update predictions for chosen machines
            self.predicts[[opp_index, m_index]] = self.model.predict([
                [
                    len(self.opp_moves),
                    self.n_pulls_self[opp_index],
                    self.n_success_self[opp_index],
                    self.n_pulls_opp[opp_index]
                ],
                [
                    len(self.opp_moves),
                    self.n_pulls_self[m_index],
                    self.n_success_self[m_index],
                    self.n_pulls_opp[m_index]
                ]])
            

def agent(observation, configuration):
    global curr_agent
    
    if observation.step == 0:
        # Initialize agent
        curr_agent = GreedyStrategy(
            "Greedy_tree",
            observation['agentIndex'],
            configuration['banditCount'])
    
    # Update payout ratio distribution with:
    curr_agent.updateDist(observation['reward'], observation['lastActions'])
    return curr_agent()