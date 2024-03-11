import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dimod.reference.samplers import ExactSolver
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler
from job_shop_scheduler import get_jss_bqm
from utils import *

# Problem Definition
# (machines,equipments,duration)

jobs = {"job_1": [([1], [1,2], 1),([2], [1], 1)],
        "job_2": [([3], [1], 1),([4], [1], 1)],
        "job_3": [([5], [1], 1)]}

# jobs = {"job_1": [(["1","3"], 2),(["3","2"], 1)],
#         "job_2": [(["3"], 3)]}

machine_downtimes = {}

timespan = 5
bqm = get_jss_bqm(jobs, machine_downtimes, timespan)

# Solve Problem

solver = "ExactSolver"
# solver = "EmbeddingComposite"

if solver == "ExactSolver":
    sampler = ExactSolver()
    sampleset = sampler.sample(bqm)

elif solver == "EmbeddingComposite":
    sampler = EmbeddingComposite(DWaveSampler())
    sampleset = sampler.sample(bqm, num_reads=1000)

solution = sampleset.first

path = "Demo/Results/"

# Save Inputs

file = open(path + "input.txt", "w")
file.write(str(jobs))
file.close()

file = open(path + "input.txt", "w")
file.write(str(bqm))
file.close()

# Save Outputs

try:
    df = solution_to_dataframe(solution.sample,jobs)
    df.to_csv(path + "solution.csv", index=False)
    print("{:<{}}".format("[SUCCESS]", 10) + "Solution saved into solution.csv")
except Exception as e:
    print("{:<{}}".format("[FAIL]", 10) + "Could not save solution into a csv file")

try:
    file = open(path + "solution.txt", "w")
    file.write(str(solution))
    file.close()
    print("{:<{}}".format("[SUCCESS]", 10) + "Solution saved as text in solution.txt")
except Exception as e:
    print("{:<{}}".format("[FAIL]", 10) + "Could not save solution into a txt file")

try:
    export_gantt_diagram("Demo/Results/","Gantt-chart",path + "solution.csv")
    print("{:<{}}".format("[SUCCESS]", 10) + "Gantt Diagram can be vizualized")
except Exception as e:
    print("{:<{}}".format("[FAIL]", 10) + "Gantt Diagram could not be build")