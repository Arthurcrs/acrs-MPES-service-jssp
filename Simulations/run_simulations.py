import sys
import os
import pandas as pd
import numpy as np
import os
import time
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from job_shop_scheduler import *
from utils import *
from Simulations.simulation_manager import *

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import LeapHybridSampler
from dwave.samplers import SimulatedAnnealingSampler
from tabu import TabuSampler
from greedy import SteepestDescentSampler

def get_simulation_managers():

    op_max_duration = 2
    max_n_machine_downtimes_intervals = 1
    n_equipments = 2

    target_variables_values = [160,320,640,1280]
    n_machines_values =       [3,  3,  4,  5   ]
    n_jobs_values =           [3,  4,  4,  5   ]
    op_per_job_values =       [3,  3,  4,  4   ]
    timespan_values   =       [16, 16, 16, 16  ]
        
    simulation_managers = []

    for i in range(len(target_variables_values)):
        simulation_manager = Simulation_Manager(
            n_jobs=n_jobs_values[i],
            n_machines=n_machines_values[i],
            n_equipments=n_equipments,
            timespan=timespan_values[i],
            op_per_job=op_per_job_values[i],
            max_n_machine_downtimes_intervals=max_n_machine_downtimes_intervals,
            op_max_duration=op_max_duration,
            target_variables=target_variables_values[i]
        )
        simulation_managers.append(simulation_manager)

    return simulation_managers

n_samples = 10
simulation_managers = get_simulation_managers()
current_datetime_as_string = get_current_datetime_as_string()
simulations_path = "Simulations/Results/sim-" + current_datetime_as_string + "/"

simulations_results = {}

for simulation_manager in simulation_managers:

    samplers_sample_times = []
    samplers_energies = []
    simulation_results = {}

    jobs = simulation_manager.jobs
    machine_downtimes = simulation_manager.machine_downtimes
    timespan = simulation_manager.timespan
    scheduler = JobShopScheduler(jobs, machine_downtimes, timespan)    
    bqm = scheduler.get_bqm()

    simulation_manager.set_simulation_directory_path(simulations_path)
    simulation_manager.set_bqm(bqm)
    simulation_manager.set_makespan_function_max_value(scheduler.makespan_function_max_value)

    simulation_manager.save_input_in_txt()
    simulation_manager.save_bqm_in_txt()

    samplers = [ 
        # ('DwaveSampler', EmbeddingComposite(DWaveSampler())),
        # ('LeapHybridSampler', LeapHybridSampler()),
        ('SimulatedAnnealing', SimulatedAnnealingSampler()),
        ('TabuSampler', TabuSampler()),
        ('SteepestDescentSampler',SteepestDescentSampler()),
    ]

    for sampler in samplers:
        
        sampler_simulation_results = {}

        sampler_title = sampler[0]
        start_time = time.time()
        sampleset = sampler[1].sample(bqm,num_reads = n_samples)
        end_time = time.time()
        sample_time = end_time - start_time

        solution = sampleset.first
        energies = sampleset.record.energy
        samplers_sample_times.append([sampler_title, sample_time])
        samplers_energies.append([sampler_title, energies])
        min_energy = np.min(energies)

        simulation_manager.set_solution(solution)
        simulation_manager.set_sample_time(sample_time)
        simulation_manager.set_energies(energies)
        simulation_manager.set_sampleset(sampleset)
        simulation_manager.set_min_energies(min_energy)
        simulation_manager.set_sampler_results_directory_path(sampler_title)

        simulation_manager.save_solution_in_csv()
        simulation_manager.create_gantt_diagram()
        simulation_manager.save_additional_info()
        simulation_manager.save_energy_occurrences_graph()
        simulation_manager.save_energy_results_in_file()
        simulation_manager.save_sampleset_to_file()

        sampler_simulation_results['solution'] = sampleset.first
        sampler_simulation_results['energies'] = sampleset.record.energy
        sampler_simulation_results['sample_time'] = sample_time
        sampler_simulation_results['sampleset'] = sampleset
        sampler_simulation_results['min_energy'] = min_energy
        sampler_simulation_results['simulation_manager'] = copy.copy(simulation_manager)
        
        if sampler == 'DwaveSampler' or sampler == 'LeapHybridSampler':
            timing_info = sampleset.info['timing']
            sampler_simulation_results['timing_info'] = timing_info
 
        simulation_results[sampler_title] = sampler_simulation_results

    simulations_results[simulation_manager.target_variables] = simulation_results
    simulation_manager.save_energy_distribution_graph_across_samplers(samplers_energies)
    simulation_manager.save_samplers_times_graph(samplers_sample_times)

save_best_solution_energy_graph(simulations_results, simulations_path)
save_sampling_time_graph(simulations_results, simulations_path)
save_dataframe_info(simulations_results, simulations_path)