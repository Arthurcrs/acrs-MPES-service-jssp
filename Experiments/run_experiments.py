import sys
import os
import pandas as pd
import numpy as np
import itertools

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from job_shop_scheduler import get_jss_bqm
from utils import *
from Experiments.experiment_manager import *

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import LeapHybridSampler
from dwave.samplers import SimulatedAnnealingSampler
from tabu import TabuSampler
from greedy import SteepestDescentSampler

samplers = [ 
    # ('DwaveSampler', EmbeddingComposite(DWaveSampler())),
    # ('LeapHybridSampler', LeapHybridSampler()),
    ('SimulatedAnnealingSampler', SimulatedAnnealingSampler()),
    ('TabuSampler', TabuSampler()),
    ('SteepestDescentSampler',SteepestDescentSampler()),
]


df_bqm_details = pd.DataFrame(columns=['Sampler',
                                       'Jobs',
                                       'Operations on each job',
                                       'Machines on each operation',
                                       'Equipments on each operation',
                                       'Variables before pruning',
                                       'BQM Variables',
                                       'BQM Interactions',
                                       'Lowest energy'
                                       ])

operation_duration = 1

experiment_method = 'sizes' # 'linear_sizes' , njobs, operations per job and machines per operation have the same value, then each following operation in a job requires a different equipment


if experiment_method == 'combination':
    n_jobs_values = [1,2,3]
    n_operations_per_job_values = [1,2,3]
    n_machines_per_operation_values = [3]
    n_equipments_per_operation_values = [1]
    entries = itertools.product(n_jobs_values, n_operations_per_job_values, n_machines_per_operation_values, n_equipments_per_operation_values)
elif experiment_method == 'sizes':
    entries = [2,3,4,5,6,7,8,9,10,11,12,13]

for entry in entries:

    if experiment_method == 'combination':
        n_jobs, n_operations_per_job, n_machines_per_operation, n_equipments_per_operation = entry
        timespan = n_jobs * n_operations_per_job
        jssp_parameters = generate_jssp_dict(n_jobs, n_operations_per_job, n_machines_per_operation, timespan, n_equipments_per_operation, operation_duration)
        combination_string = "_".join(map(str, entry))
    elif  experiment_method == 'sizes':
        n_jobs = entry 
        n_operations_per_job = entry
        n_machines_per_operation = entry
        n_equipments_per_operation =  1
        jssp_parameters = generate_jssp_dict_based_on_size(entry)
        combination_string = 'size_' + str(entry)

    for sampler in samplers:
        try:
            jobs = jssp_parameters["jobs"]
            machine_downtimes = jssp_parameters["machine_downtimes"]
            timespan = jssp_parameters["timespan"]
            
            bqm = get_jss_bqm(jobs, machine_downtimes, timespan)
            
            sampleset = sampler[1].sample(bqm,num_reads = 1000)
            energies = sampleset.record.energy
            min_energy = np.min(energies)

            solution = sampleset.first

            experiment_manager = ExperimentManager(sampler[0],jobs, machine_downtimes, timespan, bqm, sampleset, combination_string)
            experiment_manager.save_input_in_txt()
            experiment_manager.save_solution_in_csv()
            experiment_manager.create_gantt_diagram()
            experiment_manager.save_additional_info()
            experiment_manager.save_energy_occurrences_graph(energies)

            new_row = pd.DataFrame([{
                'Sampler' : sampler[0],
                'Jobs' : n_jobs,
                'Operations on each job': n_operations_per_job,
                'Machines on each operation' : n_machines_per_operation,
                'Equipments on each operation' : n_equipments_per_operation,
                'Variables before pruning': n_operations_per_job * n_jobs * n_machines_per_operation * timespan,
                'BQM Variables': bqm.num_variables,
                'BQM Interactions': bqm.num_interactions,
                'Lowest Energy': min_energy
            }])

            df_bqm_details = pd.concat([df_bqm_details, new_row], ignore_index=True)
            print("Experiment is a Success! Sampler: {}, Number of variables: {}, Number of Interactions: {}".format(sampler[0], bqm.num_variables, bqm.num_interactions))

        except Exception as e:
            print("Experiment Failed! Sampler: {}, Number of jobs: {}, Number of Interactions: {}".format(sampler[0], bqm.num_variables, bqm.num_interactions))

csv_file_path = 'Experiments/experiment_results.csv'
df_bqm_details.to_csv(csv_file_path, index=False)



