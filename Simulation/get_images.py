import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import pickle

from utils import *
from simulation_graph_manager import *

number_of_variables = 50
results_dir_path = 'Simulation/Results/' + str(number_of_variables) + '_variables' + '/'
images_dir_path = results_dir_path + 'Images/'

samplers = [
    'DwaveSampler',
    'LeapHybridSampler',
    'SimulatedAnnealing',
    'TabuSampler',
    'SteepestDescentSampler'
]

if not os.path.exists(images_dir_path):
    os.mkdir(images_dir_path)

if not os.path.exists(images_dir_path + 'Gantt Diagrams'):
    os.mkdir(images_dir_path + 'Gantt Diagrams')

with open(results_dir_path + 'sjssp.pkl', 'rb') as file:
    sjssp = pickle.load(file)

jobs = sjssp['jobs']
machine_downtimes = sjssp['machine_downtimes']
timespan = sjssp['timespan']
makespan_function_max_value = sjssp['makespan_function_max_value']

for sampler in samplers:
    try:
        new_export_gantt_diagram(images_dir_path + 'Gantt Diagrams/', 
                                sampler + '_' + str(number_of_variables) + '_gantt_chart', 
                                results_dir_path + sampler + '_solution.csv', 
                                machine_downtimes, 
                                timespan,
                                title=sampler)
    except:
        print('Could not generate the ' + sampler + ' Gantt Diagram')


simulation_graph_manager = SimulationGraphsManager(results_dir_path, images_dir_path, number_of_variables)
simulation_graph_manager.plot_lowest_energies()
simulation_graph_manager.plot_percent_of_valid_solutions()