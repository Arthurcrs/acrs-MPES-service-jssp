from __future__ import print_function
from dimod.reference.samplers import ExactSolver
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler
from job_shop_scheduler import get_jss_bqm
from utils import *

# Problem Definition

# jobs = {"job_1": [(["1","2"], 1),(["3"], 1)],
#         "job_2": [(["2"], 1),(["3","1"], 1)],
#         "job_3": [(["1"], 4)]}

jobs = {"job_1": [(["1","3"], 1),(["3","2"], 1)],
        "job_2": [(["3"], 3)]}

# machine_downtimes = {"machine_3" : [0,1,5],
#                      "machine_1" : [1,2,5]}

# Construct a BQM for the jobs

max_time = 5
bqm = get_jss_bqm(jobs, max_time)
# print(bqm)

# # Submit BQM

sampler = ExactSolver()
sampleset = sampler.sample(bqm)

# # sampler = EmbeddingComposite(DWaveSampler())
# # sampleset = sampler.sample(bqm,
# #                            num_reads=1000)

solution = sampleset.first

try:
    df = solution_to_dataframe(solution.sample,jobs)
    df.to_csv('solution.csv', index=False)
    print("{:<{}}".format("[SUCCESS]", 10) + "Solution saved into solution.csv")
except Exception as e:
    print("{:<{}}".format("[FAIL]", 10) + "Could not save solution into a csv file")

try:
    file = open("solution.txt", "w")
    file.write(str(solution))
    file.close()
    print("{:<{}}".format("[SUCCESS]", 10) + "Solution saved as text in solution.txt")
except Exception as e:
    print("{:<{}}".format("[FAIL]", 10) + "Could not save solution into a txt file")

try:
    export_gantt_diagram("image")
    print("{:<{}}".format("[SUCCESS]", 10) + "Gantt Diagram can be vizualized")
except Exception as e:
    print("{:<{}}".format("[FAIL]", 10) + "Gantt Diagram could not be build")