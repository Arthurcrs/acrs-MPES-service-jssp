import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from job_shop_scheduler import get_jss_bqm
from utils import *
from Tests.test_manager import *

from Tests.ParameterSet.tests_2jobs import tests as tests_2
from Tests.ParameterSet.tests_3jobs import tests as tests_3
from Tests.ParameterSet.tests_4jobs import tests as tests_4
from Tests.ParameterSet.tests_5jobs import tests as tests_5
from Tests.ParameterSet.tests_6jobs import tests as tests_6
from Tests.ParameterSet.tests_7jobs import tests as tests_7
from Tests.ParameterSet.tests_8jobs import tests as tests_8

from dimod.reference.samplers import ExactSolver
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler
from dwave.system.samplers import LeapHybridSampler
from dwave.samplers import SimulatedAnnealingSampler
from tabu import TabuSampler
from greedy import SteepestDescentSampler

samplers_info = [ 
    # ('DwaveSampler', EmbeddingComposite(DWaveSampler())),
    # ('LeapHybridSampler', LeapHybridSampler()),
    ('SimulatedAnnealingSampler', SimulatedAnnealingSampler()),
    ('TabuSampler', TabuSampler()),
    ('SteepestDescentSampler',SteepestDescentSampler()),
    ('ExactSolver',ExactSolver())
]

test_sizes  = [2,3,4,5,6,7,8]

tests_param = {
    2 : tests_2,
    3 : tests_3,
    4 : tests_4,
    5 : tests_5,
    6 : tests_6,
    7 : tests_7,
    8 : tests_8
}

test_logs_file_path = 'Tests/Results/logs.txt'

with open(test_logs_file_path, 'w') as file:
    file.write('test_case size solver status\n')

test_ids = get_test_ids() # Define which tests to execute in the tests_to_execute.txt file

df_bqm_details = pd.DataFrame(columns=['Test ID',
                                        'Jobs',
                                        'Operations',
                                        'Machines',
                                        'Equipments',
                                        'Timespan',
                                        'Variables',
                                        'Interactions',
                                        # 'qpu_sampling_time',
                                        # 'qpu_readout_time_per_sample',
                                        # 'qpu_access_overhead_time',
                                        # 'qpu_anneal_time_per_sample',
                                        # 'qpu_access_time',
                                        # 'qpu_programming_time',
                                        # 'qpu_delay_time_per_sample',
                                        # 'total_post_processing_time',
                                        # 'post_processing_overhead_time',
                                        'Energy : Lowest',
                                        'Energy : Highest',
                                        'Energy : Median',
                                        'Energy : Average',
                                        'Energy : Standard Deviation'  
                                       ])

errorlog = []

for test_size in test_sizes:

    tests = tests_param[test_size]

    for sampler_info in samplers_info:
        results = df_bqm_details.copy()
        sampler_name = sampler_info[0]
        sampler = sampler_info[1]
        

        for test_id in test_ids:

            jobs = tests[test_id]["jobs"]
            machine_downtimes = tests[test_id]["machine_downtimes"]
            timespan = tests[test_id]["timespan"]
            
            try:
                bqm = get_jss_bqm(jobs, machine_downtimes, timespan)

                if sampler_name == 'DwaveSampler':

                    sampleset = sampler.sample(bqm, num_reads=1000)
                else :
                    sampleset = sampler.sample(bqm)

                solution = sampleset.first
                
                test_manager = TestManager(test_id,jobs, machine_downtimes, timespan, bqm, solution, sampler_name,  test_size)

                test_manager.save_input_in_txt()
                test_manager.save_solution_in_csv()
                test_manager.create_gantt_diagram()

                try:

                    # Extract energy values
                    energies = sampleset.record.energy
                    min_energy = np.min(energies)
                    max_energy = np.max(energies)
                    median_energy = np.median(energies)
                    average_energy = np.mean(energies)
                    std_dev_energy = np.std(energies)

                    # timing_info = sampleset.info['timing']
                    energies = sampleset.record.energy

                    new_row = pd.DataFrame([{
                        'Test ID': test_id,
                        'Jobs' : len(jobs),
                        'Operations' : count_total_operations(jobs),
                        'Machines': count_unique_machines(jobs),
                        'Equipments' : count_unique_equipment(jobs), 
                        'Timespan' : timespan,
                        'Variables': bqm.num_variables,
                        'Interactions': bqm.num_interactions,
                        'num_reads': len(energies),
                        # 'qpu_sampling_time' : timing_info['qpu_sampling_time'],
                        # 'qpu_readout_time_per_sample' : timing_info['qpu_readout_time_per_sample'],
                        # 'qpu_access_overhead_time' : timing_info['qpu_access_overhead_time'],
                        # 'qpu_anneal_time_per_sample' : timing_info['qpu_anneal_time_per_sample'],
                        # 'qpu_access_time' : timing_info['qpu_access_time'],
                        # 'qpu_programming_time' :  timing_info['qpu_programming_time'],
                        # 'qpu_delay_time_per_sample' : timing_info['qpu_delay_time_per_sample'],
                        # 'total_post_processing_time' : timing_info['total_post_processing_time'],
                        # 'post_processing_overhead_time' : timing_info['post_processing_overhead_time'],
                        'Energy : Lowest': min_energy,
                        'Energy : Highest': max_energy,
                        'Energy : Median': median_energy,
                        'Energy : Average': average_energy,
                        'Energy : Standard Deviation': std_dev_energy  
                    }])

                except KeyError: # In case there is no timing info
                    new_row = pd.DataFrame([{
                        'Test ID': test_id,
                        'Jobs' : len(jobs),
                        'Operations' : count_total_operations(jobs),
                        'Machines': count_unique_machines(jobs),
                        'Equipments' : count_unique_equipment(jobs), 
                        'Timespan' : timespan,
                        'Variables': bqm.num_variables,
                        'Interactions': bqm.num_interactions,
                        'num_reads': len(energies),
                        'Energy : Lowest': min_energy,
                        'Energy : Highest': max_energy,
                        'Energy : Median': min_energy,
                        'Energy : Average': average_energy,
                        'Energy : Standard Deviation': std_dev_energy  
                    }])

                results = pd.concat([results, new_row], ignore_index=True)
                
                with open(test_logs_file_path, 'a') as file:
                    file.write(f'\n{test_id} {test_size} {sampler_name} success')
                print(f'{test_id} {test_size} {sampler_name} success')

            except Exception as e:
                with open(test_logs_file_path, 'a') as file:
                    file.write(f'\n{test_id} {test_size} {sampler_name} fail')
                print(f'{test_id} {test_size} {sampler_name} fail')

        csv_file_path = f'Tests/Results/{test_size}_operations/{sampler_name}_bqm_details.csv'
        results.to_csv(csv_file_path, index=False)