import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from job_shop_scheduler import get_jss_bqm
from utils import *
from Tests.test_manager import *
from tests import tests

from dimod.reference.samplers import ExactSolver
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler

test_ids = get_test_ids() # Define which tests to execute in the tests_to_execute.txt file

sampler = ExactSolver() # Classical sampler
# sampler = EmbeddingComposite(DWaveSampler()) # Quantum sampler

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

        print(test_id + ": Results saved")

    except Exception as e:

        print(test_id + ": Failed to get results")