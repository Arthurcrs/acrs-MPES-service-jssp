import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from Experiments.experiment_manager import *


date_of_experiment = '2024-10-28_02-37-24'
results_dir_path = "Experiments/Results/" + date_of_experiment + "/"

def redo_experiments_results(results_dir_path):
    with open(results_dir_path + 'experiment_manager.pkl', 'rb') as file:
        experiment_manager = pickle.load(file)
    experiment_manager.plot_minimum_energies_vs_variables()