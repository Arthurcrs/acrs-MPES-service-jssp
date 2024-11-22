import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import *
import pickle
import numpy as np
import pandas as pd
import time

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import LeapHybridSampler
from dwave.samplers import SimulatedAnnealingSampler
from tabu import TabuSampler
from greedy import SteepestDescentSampler


results_dir_path = 'Simulation/Results/' + '150_variables' + '/'

n_reads = 1000
samplers = [
    {'name': 'DwaveSampler', 'sampler': EmbeddingComposite(DWaveSampler())},
    {'name': 'LeapHybridSampler', 'sampler': LeapHybridSampler()},
    {'name': 'SimulatedAnnealing', 'sampler': SimulatedAnnealingSampler()},
    {'name': 'TabuSampler', 'sampler': TabuSampler()},
    {'name': 'SteepestDescentSampler', 'sampler': SteepestDescentSampler()}
]

results_columns = [
    'Amostrador',
    'Número de trabalhos',
    'Número de máquinas distintas utilizadas',
    'Número de equipamentos distintos utilizados',
    'Número de operações',
    'Timespan',
    'Número de variáveis',
    'Número de interações',
    'Menor energia',
    'Maior energia',
    'Energia média',
    'Desvio padrão de energia',
    'Mediana de energia',
    'Tempo de amostragem',
    'Percentual de soluções válidas',
    'Percentual médio de chainbreaks entre as amostras',
    'Percentual de amostras com alta quantidade de chain breaks (>15%)'
]

energy_columns = [
    'Amostrador',
    'Energia',
    'Percentual de chainbreaks'
]

results_df = pd.DataFrame(columns=results_columns)
energy_df = pd.DataFrame()

with open(results_dir_path + 'bqm.pkl', 'rb') as file:
    bqm = pickle.load(file)

with open(results_dir_path + 'sjssp.pkl', 'rb') as file:
    sjssp = pickle.load(file)

jobs = sjssp['jobs']
machine_downtimes = sjssp['machine_downtimes']
timespan = sjssp['timespan']
makespan_function_max_value = sjssp['makespan_function_max_value']

for sampler in samplers:
    start_time = time.time()
    sampleset = sampler['sampler'].sample(bqm) if sampler['name'] == 'LeapHybridSampler' else sampler['sampler'].sample(bqm, num_reads=n_reads)
    chain_breaks_data = sampleset.record.chain_break_fraction if sampler['name'] == 'DwaveSampler' else [-1] * n_reads
    end_time = time.time()
    sample_time = end_time - start_time
    solution = sampleset.first
    energies = sampleset.record.energy

    # Solution df
    solution_df = solution_to_dataframe(solution.sample,jobs)
    solution_df.to_csv(results_dir_path + sampler['name'] + '_solution.csv', index=False)

    # Append to results df
    new_row = pd.DataFrame([{
        'Amostrador': sampler['name'],
        'Número de trabalhos': len(jobs),
        'Número de máquinas distintas utilizadas': count_unique_machines(jobs),
        'Número de equipamentos distintos utilizados': count_unique_equipment(jobs),
        'Número de operações': count_total_operations(jobs),
        'Timespan': timespan,
        'Número de variáveis': bqm.num_variables,
        'Número de interações': bqm.num_interactions,
        'Menor energia': np.min(energies),
        'Maior energia': np.max(energies),
        'Energia média': np.mean(energies),
        'Desvio padrão de energia': np.std(energies),
        'Mediana de energia': np.median(energies),
        'Tempo de amostragem': sample_time,
        'Percentual de soluções válidas': get_percentage_of_valid_results(energies, makespan_function_max_value)
    }])

    if sampler['name'] == 'DwaveSampler':
        new_row['Percentual médio de chainbreaks entre as amostras'] = np.mean(chain_breaks_data) * 100
        new_row['Percentual de amostras com alta quantidade de chain breaks (>15%)'] = 100 * np.count_nonzero(chain_breaks_data > 0.15)/len(chain_breaks_data)

    results_df = pd.concat([results_df, new_row], ignore_index=True)
    
    # Append to energies df
    for energy, cb in zip(energies, chain_breaks_data):
        new_row = pd.DataFrame([{
            'Amostrador': sampler['name'],
            'Energia' : energy,
            'Percentual de chainbreaks' : cb * 100 if sampler['name'] == 'DwaveSampler' else None
        }])
        energy_df = pd.concat([energy_df, new_row], ignore_index=True)

results_df.to_csv(results_dir_path + 'results.csv', index=False)
energy_df.to_csv(results_dir_path + 'energies.csv', index=False)