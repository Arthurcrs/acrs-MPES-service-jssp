# Copyright 2019 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler
from dwave.system.samplers import LeapHybridSampler
from dwave.system.samplers import DWaveCliqueSampler

from job_shop_scheduler import get_jss_bqm, is_auxiliary_variable
from Experiments.Experiments_All import Exp_List # alterar essa importação para utilizar os diferentes arquivos.
#from Experiments_Random import Exp_List

file = open("DWave_Sampler_Data.txt", "w")
file.close()

file = open("Hybrid_Data.txt", "w")
file.close()

file = open("Clique_Result.txt", "w")
file.close()

for i in range(len(Exp_List)):
    max_time = 0
    # Construct a BQM for the jobs
    jobs = Exp_List[i]
    Exp_Name = 'Exp_{num}'.format(num = i + 1)
    for key in jobs:
        if (max_time < len(jobs[key])):
            max_time = len(jobs[key])
    #max_time = Upperbound on how long the schedule can be; alters according to the length of experiments.
    bqm = get_jss_bqm(jobs, max_time)

    sampler = EmbeddingComposite(DWaveSampler())
    sampleset = sampler.sample(bqm, chain_strength=2, num_reads=1000)
    DWave_Result = sampleset.first
    file_dwave = open("DWave_Sampler_Result.txt", "a")
    file_dwave.write(Exp_Name + " -> " + str(DWave_Result) + "\n")

    sampler = LeapHybridSampler()
    sampleset = sampler.sample(bqm, num_reads = 1000)
    Hybrid_Result = sampleset.first
    file_hybrid = open("Hybrid_Result.txt", "a")
    file_hybrid.write(Exp_Name + " -> " + str(Hybrid_Result) + "\n")

    sampler = DWaveCliqueSampler()
    sampler.largest_clique_size > 9
    sampleset = sampler.sample(bqm, num_reads = 1000)
    Clique_Result = sampleset.first
    file_clique = open("Clique_Result.txt", "a")
    file_clique.write(Exp_Name + " -> " + str(Clique_Result) + "\n")


    solution = DWave_Result.sample

    loop_count = "Loop Nº {num}".format(num = i + 1)
    print(loop_count)
    if ((i + 1) == len(Exp_List)):
        print("Concluded")
    # Grab selected nodes
    '''selected_nodes = [k for k, v in solution.items() if v == 1]

    # Parse node information
    task_times = {k: [-1]*len(v) for k, v in jobs.items()}
    for node in selected_nodes:
        if is_auxiliary_variable(node):
            continue
        job_name, task_time = node.rsplit("_", 1)
        task_index, start_time = map(int, task_time.split(","))

        task_times[job_name][task_index] = start_time

    # Print problem and restructured solution
    print("Jobs and their machine-specific tasks:")
    for job, task_list in jobs.items():
        print("{0:9}: {1}".format(job, task_list))

    print("\nJobs and the start times of each task:")
    for job, times in task_times.items():
        print("{0:9}: {1}".format(job, times))'''
file_dwave.close()
file_hybrid.close()
file_clique.close()