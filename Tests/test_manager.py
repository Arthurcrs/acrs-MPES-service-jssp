import os
from utils import *
import warnings

def get_test_ids():
    test_ids_file_path = 'Tests/tests_to_execute.txt'
    with open(test_ids_file_path, 'r') as file:
        test_ids = [line.strip() for line in file]
    return test_ids

class TestManager:
    
    def __init__(self, test_id, jobs, machine_downtimes, timespan, bqm, solution):
        self.test_id = test_id
        self.jobs = jobs
        self.machine_downtimes = machine_downtimes
        self.timespan = timespan
        self.bqm = bqm
        self.solution = solution
        self.path = "Tests/Results/" + test_id + "/"

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

        export_gantt_diagram(self.path, self.test_id + "-Gantt-chart", self.path + "solution.csv")

    def save_sampleset_info(self,sampleset_info):
        file = open(self.path + "sampleset_info.txt", "w")
        try:
            file.write(sampleset_info)
        except:
            warnings.warn('No Sampleset info')
        file.close()