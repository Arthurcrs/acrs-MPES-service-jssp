import random
import os
from utils import *
import numpy as np
import matplotlib.pyplot as plt

class Simulation_Manager:
    def __init__(self, n_jobs, n_machines, n_equipments, timespan, op_per_job, max_n_machine_downtimes_intervals, op_max_duration, target_variables):
        
        self.n_jobs = n_jobs                     
        self.n_machines = n_machines              
        self.n_equipments = n_equipments                 
        self.timespan = timespan               
        self.op_per_job = op_per_job       
        self.max_n_machine_downtimes_intervals = max_n_machine_downtimes_intervals
        self.op_max_duration = op_max_duration
        self.target_variables = target_variables
        self.n_operations = self.op_per_job * self.n_jobs

        self.jobs, self.machine_downtimes, self.generated_timespan = self.generate_random_experiment_parameters()
        
        self.num_operations = count_total_operations(self.jobs)
        self.num_machines = count_unique_machines(self.jobs)
        self.num_equipments = count_unique_equipment(self.jobs)

    def generate_random_experiment_parameters(self):
        jobs = {}
        machine_downtimes = {}
        
        # Ensure total operations is calculated by multiplying operations per job by number of jobs
        self.n_operations = self.op_per_job * self.n_jobs

        attempt_count = 0
        max_attempts = 200000  # To avoid infinite loops, set a max number of retries
        half_timespan = self.timespan // 2

        while attempt_count < max_attempts:
            jobs = {}

            # Loop through each job
            for job in range(1, int(self.n_jobs) + 1):
                operations = []
                num_operations = self.op_per_job  # Fixed number of operations per job
                
                for _ in range(num_operations):
                    # Random machines for the operation
                    machines = random.sample(range(1, self.n_machines + 1), random.randint(1, self.n_machines))

                    # Random equipment for the operation
                    equipment = random.sample(range(1, self.n_equipments + 1), random.randint(0, self.n_equipments))

                    # Random duration for the operation (between 1 and op_max_duration)
                    duration = random.randint(1, self.op_max_duration)

                    operations.append((machines, equipment, duration))

                jobs[f"job_{job}"] = operations

            # Check the total number of variables
            total_variables = calculate_total_variables(jobs, self.timespan)

            # If target_variables is not specified, accept any result; otherwise, ensure the correct number of variables
            if self.target_variables is None or total_variables == self.target_variables:
                break  # Stop when we hit the desired number of variables
            attempt_count += 1

        if attempt_count == max_attempts:
            print(f"Warning: Could not generate exactly {self.target_variables} variables after {max_attempts} attempts.")

        # Generate random downtimes for machines (always in the first half of the timespan)
        for machine in range(1, self.n_machines + 1):
            num_downtimes = random.randint(1, self.max_n_machine_downtimes_intervals)
            downtimes = sorted(random.sample(range(half_timespan), num_downtimes))
            machine_downtimes[machine] = downtimes

        return jobs, machine_downtimes, self.timespan
   
    def set_simulation_directory_path(self, current_datetime_as_string):
        self.simulation_directory_path = "Simulations/Results/sim-" + current_datetime_as_string + "/"
        os.makedirs(self.simulation_directory_path, exist_ok=True)
        self.variable_directory_path = self.simulation_directory_path + str(self.target_variables) + "-variables/"
        os.makedirs(self.variable_directory_path, exist_ok=True)
            
    def set_sampler_results_directory_path(self, sampler_title):
        self.sampler_title = sampler_title
        self.sampler_results_directory_path = self.variable_directory_path + sampler_title + "/"
        os.makedirs(self.sampler_results_directory_path, exist_ok=True)

    def set_bqm(self, bqm):
        self.bqm = bqm

    def set_makespan_function_max_value(self,makespan_function_max_value):
        self.makespan_function_max_value = makespan_function_max_value

    def set_solution(self, solution):
        self.solution = solution

    def set_energies(self,energies):
        self.energies = energies

    def set_min_energies(self, min_energies):
        self.min_energy = min_energies

    def set_sampleset(self, sampleset):
        self.sampleset = sampleset
    
    def set_sample_time(self, sample_time):
        self.sample_time = sample_time

    def save_input_in_txt(self):

        file = open(self.variable_directory_path + "inputs.txt", "w")
        file.write("Jobs:\n" + str(self.jobs) + "\n\n")
        file.write("Machine Downtimes:\n" + str(self.machine_downtimes) + "\n\n")
        file.write("Timespan:\n" + str(self.timespan))
        file.close()

    def save_bqm_in_txt(self):
        file = open(self.variable_directory_path + "bqm.txt", "w")
        file.write(str(self.bqm))
        file.close()

    def save_solution_in_csv(self):

        df = solution_to_dataframe(self.solution.sample,self.jobs)
        df.to_csv(self.sampler_results_directory_path + "solution.csv", index=False)

    def save_energy_occurrences_graph(self):
        plt.figure(figsize=(10, 6))
        
        # Use logarithmic bins for large variations in energy values
        bins = np.logspace(np.log10(min(self.energies)), np.log10(max(self.energies)), 20)
        
        # Plotting the histogram
        plt.hist(self.energies, bins=bins, edgecolor='black', alpha=0.7)
        
        # Set the x-axis to log scale for better distribution visualization
        plt.xscale('log')
        
        # Adding labels and title
        plt.xlabel('Energia')
        plt.ylabel('Frequência')
        plt.title('Distribuição de energias')

        # Add a vertical line at the threshold for invalid results
        plt.axvline(x=self.makespan_function_max_value, color='red', linestyle='--', linewidth=2, label=f'Limite para resultados válidos')
        
        # Add a legend for the vertical line
        plt.legend()

        # Save the plot to the specified path
        plt.savefig(self.sampler_results_directory_path + "energy_distribution")
        
        # Close the plot to free memory
        plt.close()

    def save_sampleset_to_file(self):
        file = open(self.sampler_results_directory_path + "sampleset.txt", "w")
        file.write(str(self.sampleset))
        file.close()

    def create_gantt_diagram(self):
        new_export_gantt_diagram(self.sampler_results_directory_path, self.sampler_title + "-Gantt-chart", self.sampler_results_directory_path + "solution.csv", self.machine_downtimes, self.timespan)

    def save_additional_info(self):
        file = open(self.sampler_results_directory_path + "additional.txt", "w")
        file.write("Number of variables before prunning: {} \n".format(calculate_total_variables(self.jobs,self.timespan)))
        file.write("BQM - Number of variables: {} \n".format(self.bqm.num_variables))
        file.write("BQM - Number of interactions: {} \n".format(self.bqm.num_interactions))
        file.write("Number of jobs: {} \n".format(self.n_jobs))
        file.write("Total number of operations: {} \n".format(self.num_operations))
        file.write("Number of unique machines: {} \n".format(self.num_machines))
        file.write("Number of unique equipments: {} \n".format(self.num_equipments))
        file.write("Energy: {} \n".format(self.min_energy))
        file.write("Percentage of valid solutions: {} \n".format(self.get_percentage_of_valid_results()))
        file.write("Sample Time: {} \n".format(self.sample_time))

        if self.makespan_function_max_value > self.min_energy:
            file.write("Is solution valid?: True")
        else:
            file.write("Is solution valid?: False")

        file.close()

    def get_percentage_of_valid_results(self):
        total_count = len(self.energies)
        valid_count = np.sum(self.energies <= self.makespan_function_max_value)
        valid_percentage = (valid_count / total_count) * 100
        return valid_percentage
    
    def save_energy_results_in_file(self):
        np.savetxt(self.sampler_results_directory_path + "energy_array", self.energies, delimiter=',', fmt='%f')

    def save_energy_distribution_graph_across_samplers(self, samplers_energies):
        plt.figure(figsize=(10, 6))
        
        # Prepare a list for all energies and labels
        all_energies = []
        labels = []
        
        # Iterate through each sampler to prepare the data
        for sampler_title, energies in samplers_energies:
            all_energies.append(energies)
            labels.append(sampler_title)
        
        # Use logarithmic bins for large variations in energy values
        bins = np.logspace(np.log10(min([min(e) for e in all_energies])), 
                        np.log10(max([max(e) for e in all_energies])), 20)
        
        # Plot the stacked histogram
        plt.hist(all_energies, bins=bins, label=labels, edgecolor='black', linewidth=1.2, stacked=True)

        # Set the x-axis to log scale for better distribution visualization
        plt.xscale('log')
        
        # Add a vertical line at the threshold for invalid results
        plt.axvline(x=self.makespan_function_max_value, color='red', linestyle='--', linewidth=2, label=f'Limite para resultados válidos')
        
        # Adding labels and title
        plt.xlabel('Energia')
        plt.ylabel('Frequência')
        plt.title('Distribuição de energias (' + str(self.target_variables) + ' variáveis)')
        
        # Add a legend and move it outside the plot
        plt.legend(title="Amostradores", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust the layout to make room for the legend
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to fit the legend outside the plot
        
        # Save the plot to the specified path
        plt.savefig(self.variable_directory_path + "energy_distribution_across_samplers.png", bbox_inches='tight')
        
        # Close the plot to free memory
        plt.close()

    def save_samplers_times_graph(self, samplers_sample_times):
        # Extract titles and times
        sampler_titles = [item[0] for item in samplers_sample_times]
        sample_times = [item[1] for item in samplers_sample_times]

        # Create a bar chart to represent sampling times
        plt.figure(figsize=(10, 6))
        plt.barh(sampler_titles, sample_times, color='skyblue', edgecolor='black')

        # Adding labels and title
        plt.xlabel('Tempo (segundos)')
        plt.ylabel('Amostradores')
        plt.title('Tempo de amostragem')

        # Adding values on the bars
        for index, value in enumerate(sample_times):
            plt.text(value + (max(sample_times) * 0.03), index, f'{value:.2f} s', va='center')  # Added space by shifting text
        
        # Add some extra space on the x-axis for the text
        plt.xlim([0, max(sample_times) * 1.2])  # Extend x-axis to give more room for the text

        # Adjust layout for better presentation
        plt.tight_layout()

        # Save the plot to the specified path
        plt.savefig(self.variable_directory_path + "samplers_sample_times.png")
        
        # Close the plot to free memory
        plt.close()