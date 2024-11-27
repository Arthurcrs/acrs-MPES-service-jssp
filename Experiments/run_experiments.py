import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import *
from Experiments.experiment_manager import *
import pickle

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import LeapHybridSampler
from dwave.samplers import SimulatedAnnealingSampler
from tabu import TabuSampler
from greedy import SteepestDescentSampler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

results_dir_path = "Experiments/Results/" + get_current_datetime_as_string() + "/"

samplers = [
    {'name': 'DwaveSampler', 'sampler': EmbeddingComposite(DWaveSampler())},
    {'name': 'LeapHybridSampler', 'sampler': LeapHybridSampler()},
    {'name': 'SimulatedAnnealing', 'sampler': SimulatedAnnealingSampler()},
    {'name': 'TabuSampler', 'sampler': TabuSampler()},
    {'name': 'SteepestDescentSampler', 'sampler': SteepestDescentSampler()}
]

parameters_ranges = {
    'n_jobs':                 [2,4],
    'max_op_in_job':          [2,4], 
    'n_possible_machines':    [2,4], 
    'n_possible_equipments':  [0,2], 
}

n_reads = 50
repetitions = 1
max_operation_duration = 2

experiment_manager = ExperimentManager(n_reads, repetitions, max_operation_duration, samplers, parameters_ranges, results_dir_path)
experiment_manager.execute_experiments()
experiment_manager.export_results_to_csv()

with open(results_dir_path + 'experiment_manager.pkl', 'wb') as file:
    pickle.dump(experiment_manager, file)

experiment_manager.plot_minimum_energies_vs_variables()



