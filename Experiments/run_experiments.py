import sys
import os
import pandas as pd
import numpy as np

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

exception_list = []

samplers = [ 
    ('DwaveSampler', EmbeddingComposite(DWaveSampler())),
    ('LeapHybridSampler', LeapHybridSampler()),
    ('SimulatedAnnealingSampler', SimulatedAnnealingSampler()),
    ('TabuSampler', TabuSampler()),
    ('SteepestDescentSampler',SteepestDescentSampler())
]


n_jobs = 2
n_operations_per_job = 2
n_machines_per_operation = n_jobs * n_operations_per_job
operation_duration = 1
timespan = n_jobs * n_operations_per_job * operation_duration
n_equipments_per_operation = 1

jssp_parameters = generate_jssp_dict(n_jobs, n_operations_per_job, n_machines_per_operation, timespan, n_equipments_per_operation, operation_duration)

df_bqm_details = pd.DataFrame(columns=['Sampler',
                                       'Variables',
                                       'Interactions',
                                       'Lowest Energy'
                                       ])

for sampler in samplers:
    try:
        jobs = jssp_parameters["jobs"]
        machine_downtimes = jssp_parameters["machine_downtimes"]
        timespan = jssp_parameters["timespan"]
        
        bqm = get_jss_bqm(jobs, machine_downtimes, timespan)
        
        sampleset = sampler[1].sample(bqm)
        energies = sampleset.record.energy
        min_energy = np.min(energies)

        solution = sampleset.first

        experiment_manager = ExperimentManager(sampler[0],jobs, machine_downtimes, timespan, bqm, solution)
        experiment_manager.save_input_in_txt()
        experiment_manager.save_solution_in_csv()
        experiment_manager.create_gantt_diagram()
        experiment_manager.save_additional_info(min_energy)

        new_row = pd.DataFrame([{
            'Sampler' : sampler[0],
            'Variables': bqm.num_variables,
            'Interactions': bqm.num_interactions,
            'Lowest Energy': min_energy
        }])

        df_bqm_details = pd.concat([df_bqm_details, new_row], ignore_index=True)
        print("Experiment is a Success! Sampler: {}, Number of variables: {}, Number of Interactions: {}".format(sampler[0], bqm.num_variables, bqm.num_interactions))

    except Exception as e:
        print("Experiment Failed! Sampler: {}, Number of jobs: {}, Number of Interactions: {}".format(sampler[0], bqm.num_variables, bqm.num_interactions))
        if e not in exception_list:
            exception_list.append(e)


for exceptions in exception_list:
    print('Exception with Sampler: {} : {}'.format(sampler[0],exceptions))



