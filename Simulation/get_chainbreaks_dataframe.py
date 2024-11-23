import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import *
import pickle
import numpy as np
import pandas as pd

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

results_dir_path = 'Simulation/Results/' + '150_variables' + '/'

n_reads = 1000  

if not os.path.exists(results_dir_path + 'Chain Strength Tests'):
    os.mkdir(results_dir_path + 'Chain Strength Tests')

chain_strength_tests_columns = [
    'Chain Strength',
    'Menor energia',
    'Percentual de soluções válidas',
    'Percentual médio de chainbreaks entre as amostras',
    'Percentual de amostras sem chainbreaks',
    'Percentual de amostras com alta quantidade de chain breaks (>15%)'
]

energy_columns = [
    'Chain Strength',
    'Energia',
    'Percentual de chainbreaks'
]

result_df = pd.DataFrame(columns=chain_strength_tests_columns)
energies_df = pd.DataFrame(columns=energy_columns)

with open(results_dir_path + 'bqm.pkl', 'rb') as file:
    bqm = pickle.load(file)

with open(results_dir_path + 'sjssp.pkl', 'rb') as file:
    sjssp = pickle.load(file)

jobs = sjssp['jobs']
machine_downtimes = sjssp['machine_downtimes']
timespan = sjssp['timespan']
makespan_function_max_value = sjssp['makespan_function_max_value']

# 37748736

chain_strength_values = [
                        1,
                        makespan_function_max_value * 0.2,
                        makespan_function_max_value * 0.4,
                        makespan_function_max_value * 0.6,
                        makespan_function_max_value * 0.8,
                        makespan_function_max_value * 1.0,
                        makespan_function_max_value * 1.2,
                        makespan_function_max_value * 1.4,
                        makespan_function_max_value * 1.6,
                        makespan_function_max_value * 2.0
                        ]


for chain_strength_value in chain_strength_values:
    sampleset = EmbeddingComposite(DWaveSampler(chain_strength=chain_strength_value)).sample(bqm, num_reads=n_reads)
    chain_breaks_data = sampleset.record.chain_break_fraction
    solution = sampleset.first
    energies = sampleset.record.energy

    new_row = pd.DataFrame([{
        'Chain Strength' : chain_strength_value,
        'Menor energia': np.min(energies),
        'Percentual de soluções válidas': get_percentage_of_valid_results(energies, makespan_function_max_value),
        'Percentual médio de chainbreaks entre as amostras': np.mean(chain_breaks_data) * 100,
        'Percentual de amostras sem chainbreaks': (np.sum(chain_breaks_data == 0) / len(chain_breaks_data)) * 100,
        'Percentual de amostras com alta quantidade de chain breaks (>15%)' : 100 * np.count_nonzero(chain_breaks_data > 0.15)/len(chain_breaks_data)
    }])

    result_df = pd.concat([result_df, new_row], ignore_index=True)
    
    for energy, cb in zip(energies, chain_breaks_data):
        new_row = pd.DataFrame([{
            'Chain Strength' : chain_strength_value,
            'Energia' : energy,
            'Percentual de chainbreaks' : cb * 100
        }])
        energies_df = pd.concat([energies_df, new_row], ignore_index=True)

result_df.to_csv(results_dir_path + 'Chain Strength Tests/' + 'chain_strength_test.csv', index=False)
energies_df.to_csv(results_dir_path + 'Chain Strength Tests/' + 'chain_strength_test_energies.csv', index=False)