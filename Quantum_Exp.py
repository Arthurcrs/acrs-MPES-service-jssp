from __future__ import print_function

from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler
from dimod.reference.samplers import ExactSolver

from job_shop_scheduler import get_jss_bqm, is_auxiliary_variable
from Experiments.Exps_1_Jobs import Exp_1
#TODO: Import all Exp from the Experiment file
#TODO: Add Machine Uptime

Exp_List = [Exp_1]

for i in range(len(Exp_List)):
    
    jobs = Exp_List[i]["jobs"]
    machine_downtimes = Exp_List[i]["machine_downtimes"]

    Exp_Name = 'Exp_{num}'.format(num = i + 1)
    max_time = 7
    bqm = get_jss_bqm(jobs, machine_downtimes, max_time)
 
    sampler = ExactSolver()
    sampleset = sampler.sample(bqm)

    # sampler = EmbeddingComposite(DWaveSampler())
    # sampleset = sampler.sample(bqm, chain_strength=2, num_reads=1000)


    DWave_Result = sampleset.first
    file_dwave = open("DWave_Sampler_Result.txt", "a")
    file_dwave.write(Exp_Name + " -> " + str(DWave_Result) + "\n")

    solution = DWave_Result.sample

    if ((i + 1) == len(Exp_List)):
        print("Concluded")

file_dwave.close()