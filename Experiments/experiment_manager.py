import os
from utils import *
import numpy as np

def get_test_ids():
    test_ids_file_path = 'Tests/tests_to_execute.txt'
    with open(test_ids_file_path, 'r') as file:
        test_ids = [line.strip() for line in file]
    return test_ids

class ExperimentManager:
    
    def __init__(self, sampler_title, jobs, machine_downtimes, timespan, bqm, sampleset, results_directory_name):
        self.sampler_title = sampler_title
        self.jobs = jobs
        self.machine_downtimes = machine_downtimes
        self.timespan = timespan
        self.bqm = bqm
        self.solution = sampleset.first
        self.path = "Experiments/Results/" + results_directory_name + "/" + sampler_title + "/"
        self.num_jobs = len(self.jobs)
        self.num_operations = count_total_operations(self.jobs)
        self.num_machines = count_unique_machines(self.jobs)
        self.num_equipments = count_unique_equipment(self.jobs)
        self.num_variables = self.num_operations * self.num_machines * self.timespan
        self.min_energy = np.min(sampleset.record.energy)

        self.create_directories()

    def create_directories(self):
        os.makedirs(self.path, exist_ok=True)

    def save_input_in_txt(self):

        file = open(self.path + "inputs.txt", "w")
        file.write("Jobs:\n" + str(self.jobs) + "\n\n")
        file.write("Machine Downtimes:\n" + str(self.machine_downtimes) + "\n\n")
        file.write("Timespan:\n" + str(self.timespan))
        file.close()

        file = open(self.path + "bqm.txt", "w")
        file.write(str(self.bqm))
        file.close()

    def save_solution_in_csv(self):

        df = solution_to_dataframe(self.solution.sample,self.jobs)
        df.to_csv(self.path + "solution.csv", index=False)

    def create_gantt_diagram(self):

        export_gantt_diagram(self.path, self.sampler_title + "-Gantt-chart", self.path + "solution.csv")

    def save_additional_info(self):
        file = open(self.path + "additional.txt", "w")
        file.write("BQM - Number of variables: {} \n".format(self.bqm.num_variables))
        file.write("BQM - Number of interactions: {} \n".format(self.bqm.num_interactions))
        file.write("Number of jobs: {} \n".format(self.num_jobs))
        file.write("Total number of operations: {} \n".format(self.num_operations))
        file.write("Number of unique machines: {} \n".format(self.num_machines))
        file.write("Number of unique equipments: {} \n".format(self.num_equipments))
        file.write("Initial number of variables before trimming: {}".format(self.num_variables))
        file.write("Energy: {} \n".format(self.min_energy))
        file.close()