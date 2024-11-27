import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
from job_shop_scheduler import *
from utils import *
import random
import math

def generate_sjssp(parameters):
    jobs = {}
    machine_downtimes = {}
    timespan = -1
    
    n_jobs = parameters['n_jobs']
    n_possible_machines = parameters['n_possible_machines']
    max_op_in_job = parameters['max_op_in_job']
    n_possible_equipments = parameters['n_possible_equipments']
    max_operation_duration = 2

    for job_id in range(1, n_jobs + 1):
        job_name = f"job_{job_id}"
        n_operations = random.randint(1, max_op_in_job)
        operations = []

        for _ in range(n_operations):
            machines = random.sample(range(1, n_possible_machines + 1), random.randint(1, n_possible_machines))
            if n_possible_equipments != 0:
                equipment = [random.randint(1, n_possible_equipments)] if random.choice([True, False]) else []
            else:
                equipment = []
            duration = random.randint(1, max_operation_duration)
            operations.append((machines, equipment, duration))

        jobs[job_name] = operations

    num_operations = count_total_operations(jobs)
    num_machines = count_unique_machines(jobs)
    min_timespan = 1 + num_operations // num_machines
    max_timespan = 1 + num_operations * 2
    timespan_reduction_factor = 0.75
    timespan = math.ceil((min_timespan + max_timespan * timespan_reduction_factor)/2)
    first_half_timespan = timespan // 2
    for machine_id in range(1, n_possible_machines + 1):
        downtime_instant = random.randint(0, first_half_timespan - 1)
        machine_downtimes[machine_id] = [downtime_instant]

    return jobs, machine_downtimes, timespan

def export_bqm(bqm, results_dir_path):
    with open(results_dir_path + 'bqm.pkl', 'wb') as file:
        pickle.dump(bqm, file)

def export_sjssp(jobs, machine_downtimes, timespan, makespan_function_max_value, results_dir_path):

    sjssp = {
        'jobs' : jobs,
        'machine_downtimes' : machine_downtimes,
        'timespan' : timespan,
        'makespan_function_max_value' : makespan_function_max_value
    }

    with open(results_dir_path + 'sjssp.pkl', 'wb') as file:
        pickle.dump(sjssp, file)

desired_variables_after_trim = 50
max_number_of_attempts = 1000
parameters = {
    'n_jobs' : 3,
    'max_op_in_job': 3,
    'n_possible_machines': 2,
    'n_possible_equipments': 2
}
results_dir_path = 'Simulation/Results/' + str(desired_variables_after_trim) + '_variables_' + get_current_datetime_as_string() + '/'
os.mkdir(results_dir_path)

number_of_variables_after_trim = -1
attempts = 0
while (number_of_variables_after_trim != desired_variables_after_trim and attempts < max_number_of_attempts):

    
    jobs, machine_downtimes, timespan = generate_sjssp(parameters)
    scheduler = JobShopScheduler(jobs, machine_downtimes, timespan)
    bqm = scheduler.get_bqm()
    makespan_function_max_value = scheduler.makespan_function_max_value
    number_of_variables_after_trim = bqm.num_variables
    
    attempts = attempts + 1

if attempts == max_number_of_attempts:
    print('Could not generate a BQM with the desired number of variables after ', str(max_number_of_attempts), ' attempts')

export_bqm(bqm, results_dir_path)
export_sjssp(jobs, machine_downtimes, timespan, makespan_function_max_value, results_dir_path)