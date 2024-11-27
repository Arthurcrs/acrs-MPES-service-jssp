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

with open(results_dir_path + 'sjssp.pkl', 'rb') as file:
    sjssp = pickle.load(file)


export_sjssp_as_text(sjssp, results_dir_path)