import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from job_shop_scheduler import get_jss_bqm
from utils import *
from Tests.test_manager import *
from tests import tests

from dimod.reference.samplers import ExactSolver
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler

test_ids = get_test_ids() # Define which tests to execute in the tests_to_execute.txt file

# sampler = ExactSolver() # Classical sampler
sampler = EmbeddingComposite(DWaveSampler()) # Quantum sampler

csv_file_path = 'bqm_details.csv'
df_bqm_details = pd.DataFrame(columns=['Test ID',
                                       'Jobs',
                                       'Operations',
                                       'Machines',
                                       'Equipments',
                                       'Timespan',
                                       'Variables',
                                       'Interactions',
                                        'qpu_sampling_time',
                                        'qpu_readout_time_per_sample',
                                        'qpu_access_overhead_time',
                                        'qpu_anneal_time_per_sample',
                                        'qpu_access_time',
                                        'qpu_programming_time',
                                        'qpu_delay_time_per_sample',
                                        'total_post_processing_time',
                                        'post_processing_overhead_time'
                                       ])

for test_id in test_ids:

    jobs = tests[test_id]["jobs"]
    machine_downtimes = tests[test_id]["machine_downtimes"]
    timespan = tests[test_id]["timespan"]
    
    try:
        bqm = get_jss_bqm(jobs, machine_downtimes, timespan)
        sampleset = sampler.sample(bqm)
        solution = sampleset.first
        
        test_manager = TestManager(test_id,jobs, machine_downtimes, timespan, bqm, solution)

        test_manager.save_input_in_txt()
        test_manager.save_solution_in_csv()
        test_manager.create_gantt_diagram()

        timing_info = sampleset.info['timing']

        new_row = pd.DataFrame([{
            'Test ID': test_id,
            'Jobs' : len(jobs),
            'Operations' : count_total_operations(jobs),
            'Machines': count_unique_machines(jobs),
            'Equipments' : count_unique_equipment(jobs), 
            'Timespan' : timespan,
            'Variables': bqm.num_variables,
            'Interactions': bqm.num_interactions,
            'qpu_sampling_time' : timing_info['qpu_sampling_time'],
            'qpu_readout_time_per_sample' : timing_info['qpu_readout_time_per_sample'],
            'qpu_access_overhead_time' : timing_info['qpu_access_overhead_time'],
            'qpu_anneal_time_per_sample' : timing_info['qpu_anneal_time_per_sample'],
            'qpu_access_time' : timing_info['qpu_access_time'],
            'qpu_programming_time' :  timing_info['qpu_programming_time'],
            'qpu_delay_time_per_sample' : timing_info['qpu_delay_time_per_sample'],
            'total_post_processing_time' : timing_info['total_post_processing_time'],
            'post_processing_overhead_time' : timing_info['post_processing_overhead_time']
        }])

        df_bqm_details = pd.concat([df_bqm_details, new_row], ignore_index=True)
        
        print(test_id + ": Results saved")

    except Exception as e:

        print(test_id + ": Failed to get results")
        
csv_file_path = 'Tests/bqm_details.csv'
df_bqm_details.to_csv(csv_file_path, index=False)