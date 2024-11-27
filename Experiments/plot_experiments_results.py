import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from Experiments.experiment_manager import *


date_of_experiment = 'final'
results_dir_path = "Experiments/Results/" + date_of_experiment + "/"
sampler_colors = {
            'DwaveSampler': '#FF0000',  
            'LeapHybridSampler': '#0000FF',  
            'SimulatedAnnealing': '#006400',  
            'TabuSampler': '#8B4513',  
            'SteepestDescentSampler': '#FF00FF'  
        }

with open(results_dir_path + 'experiment_manager.pkl', 'rb') as file:
    experiment_manager = pickle.load(file)

results_df = pd.read_csv(results_dir_path + 'results.csv')

experiment_manager.set_results_path(results_dir_path)
experiment_manager.set_results_df(results_df)
experiment_manager.set_sampler_colors(sampler_colors)
experiment_manager.plot_minimum_energies_vs_variables()
experiment_manager.plot_sampling_time_vs_variables()
experiment_manager.plot_best_sampler_per_test()
experiment_manager.plot_best_sampler_percentage_per_bin()
experiment_manager.plot_average_chain_breaks_vs_variables()
experiment_manager.plot_variables_vs_interactions()
experiment_manager.plot_sampler_ranks_per_bin()
experiment_manager.plot_best_sampler_percentage_per_bin_closer()
experiment_manager.plot_high_chain_break_percentage_vs_variables()
experiment_manager.plot_operations_vs_machines()
experiment_manager.plot_operations_vs_machines_heatmap()