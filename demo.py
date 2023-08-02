
from __future__ import print_function

from dimod.reference.samplers import ExactSolver

from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler

from job_shop_scheduler import get_jss_bqm, is_auxiliary_variable



# Problem Definition

jobs = {"job_1": [("machine_1", 2), ("machine_2", 1)],
        "job_2": [("machine_2", 1)],
        "job_3": [("machine_3", 2)]}

machine_downtimes = {"machine_3" : [0,1,4,5],
                     "machine_1" : [1,2,4]}

# TODO: machine_uptimes = {}

# Construct a BQM for the jobs

max_time = 6
bqm = get_jss_bqm(jobs,machine_downtimes, max_time)

# # Submit BQM

# sampler = ExactSolver()
# sampleset = sampler.sample(bqm)

# # sampler = EmbeddingComposite(DWaveSampler())
# # sampleset = sampler.sample(bqm,
# #                            chain_strength=2,
# #                            num_reads=1000,
# #                            label='Example - Job Shop Scheduling')

# file = open('Results', 'w')
# solution = sampleset.first.sample
# file.write(str(solution))

# selected_nodes = [k for k, v in solution.items() if v == 1]

# # Parse node information
# task_times = {k: [-1]*len(v) for k, v in jobs.items()}
# for node in selected_nodes:
#     if is_auxiliary_variable(node):
#         continue
#     job_name, task_time = node.rsplit("_", 1)
#     task_index, start_time = map(int, task_time.split(","))

#     task_times[job_name][task_index] = start_time

# # Print problem and restructured solution
# print("Jobs and their machine-specific tasks:")
# for job, task_list in jobs.items():
#     print("{0:9}: {1}".format(job, task_list))

# print("\nJobs and the start times of each task:")
# for job, times in task_times.items():
#     print("{0:9}: {1}".format(job, times))

# print("\nMachines and their downtimes")
# print(machine_downtimes)
