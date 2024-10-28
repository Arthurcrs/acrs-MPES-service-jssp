import os
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from job_shop_scheduler import *
from utils import *
import random
import math
from itertools import product
import time
from tqdm import tqdm

class ExperimentManager:
    
    def __init__(self, n_reads, repetitions, max_operation_duration, samplers, parameters_ranges, results_dir_path):
        self.n_reads = n_reads
        self.repetitions = repetitions
        self.max_operation_duration = max_operation_duration
        self.samplers = samplers
        self.parameter_ranges = parameters_ranges
        self.results_dir_path = results_dir_path
        self.parameter_combinations = self.generate_parameter_combinations()
        self.init_results_df()

    def init_results_df(self):
        columns = [
            "Amostrador",
            "Número de trabalhos",
            "Número de máquinas distintas utilizadas",
            "Número de equipamentos distintos utilizados",
            "Número de operações",
            "Timespan",
            "Número de variáveis",
            "Número de interações",
            "Menor Energia",
            "Tempo de amostragem",
            "Percentual de soluções válidas",
            "Parâmetro: Número de trabalhos",
            "Parâmetro: Máximo número de operações em um trabalho",
            "Parâmetro: Número de máquinas",
            "Parâmetro: Número de equipamentos possíveis"
        ]
        self.results_df = pd.DataFrame(columns=columns)

    def generate_parameter_combinations(self):
        param_names = list(self.parameter_ranges.keys())
        param_values = [range(r[0], r[1] + 1) for r in self.parameter_ranges.values()]
        combinations = list(product(*param_values))

        return [dict(zip(param_names, combo)) for combo in combinations]

    def execute_experiments(self):
        experiment_start_time = time.time()
        with tqdm(total=len(self.parameter_combinations)) as pbar:
            for parameters in self.parameter_combinations:
                for i in range(self.repetitions):
                    jobs, machine_downtimes, timespan = self.generate_sjssp(parameters)
                    scheduler = JobShopScheduler(jobs, machine_downtimes, timespan)
                    bqm = scheduler.get_bqm()
                    makespan_function_max_value = scheduler.makespan_function_max_value
                    for sampler in self.samplers:
                        start_time = time.time()
                        if sampler['name'] == 'LeapHybridSampler':
                            sampleset = sampler['sampler'].sample(bqm)
                        else:
                            sampleset = sampler['sampler'].sample(bqm,num_reads = self.n_reads)
                        end_time = time.time()
                        sample_time = end_time - start_time
                        energies = sampleset.record.energy
                        self.add_results_to_df(sampler['name'], sample_time, jobs, timespan, bqm, energies, makespan_function_max_value, parameters)
                pbar.update(1)
        experiment_end_time = time.time()
        print('Experiments finished in: ' + str((experiment_end_time - experiment_start_time)/60) + 'minutes')

    def generate_sjssp(self, parameters):
        jobs = {}
        machine_downtimes = {}
        timespan = -1
        
        n_jobs = parameters['n_jobs']
        n_possible_machines = parameters['n_possible_machines']
        max_op_in_job = parameters['max_op_in_job']
        n_possible_equipments = parameters['n_possible_equipments']
        max_operation_duration = self.max_operation_duration

        # Generate jobs
        for job_id in range(1, n_jobs + 1):
            job_name = f"job_{job_id}"
            n_operations = random.randint(1, max_op_in_job)
            operations = []

            for _ in range(n_operations):
                # Choose a random set of machines for this operation
                machines = random.sample(range(1, n_possible_machines + 1), random.randint(1, n_possible_machines))
                # Choose a random equipment or none
                if n_possible_equipments != 0:
                    equipment = [random.randint(1, n_possible_equipments)] if random.choice([True, False]) else []
                else:
                    equipment = []
                # Set a random duration from 1 to max_operation_duration
                duration = random.randint(1, max_operation_duration)

                # Append the operation to the job's operation list
                operations.append((machines, equipment, duration))

            # Add the job to the jobs dictionary
            jobs[job_name] = operations

        # Calculate timespan
        num_operations = count_total_operations(jobs)
        num_machines = count_unique_machines(jobs)
        min_timespan = 1 + num_operations // num_machines
        max_timespan = 1 + num_operations * 2
        timespan = math.ceil((min_timespan + max_timespan)/2)

        # Generate machine downtimes
        first_half_timespan = timespan // 2
        for machine_id in range(1, n_possible_machines + 1):
            downtime_instant = random.randint(0, first_half_timespan - 1)
            machine_downtimes[machine_id] = [downtime_instant]

        return jobs, machine_downtimes, timespan

    def add_results_to_df(self, sampler_name, sample_time, jobs, timespan, bqm, energies, makespan_function_max_value, parameters):
        # Create a new row as a DataFrame
        new_row = pd.DataFrame([{
            'Amostrador': sampler_name,
            'Número de trabalhos': len(jobs),
            'Número de máquinas distintas utilizadas': count_unique_machines(jobs),
            'Número de equipamentos distintos utilizados': count_unique_equipment(jobs),
            'Número de operações': count_total_operations(jobs),
            'Timespan': timespan,
            'Número de variáveis': bqm.num_variables,
            'Número de interações': bqm.num_interactions,
            'Menor Energia': np.min(energies),
            'Tempo de amostragem': sample_time,
            'Percentual de soluções válidas': get_percentage_of_valid_results(energies, makespan_function_max_value),
            'Parâmetro: Número de trabalhos': parameters['n_jobs'],
            'Parâmetro: Máximo número de operações em um trabalho': parameters['max_op_in_job'],
            'Parâmetro: Número de máquinas': parameters['n_possible_machines'],
            'Parâmetro: Número de equipamentos possíveis': parameters['n_possible_equipments'],
        }])
        
        self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)

    def export_results_to_csv(self):
        os.makedirs(self.results_dir_path, exist_ok=True)
        self.results_df.to_csv(self.results_dir_path + 'results.csv', index=False)

    def plot_minimum_energies_vs_variables(self):
        # Define colors based on 'Amostrador' categories
        amostrador_categories = self.results_df['Amostrador'].astype('category')
        colors = amostrador_categories.cat.codes  # Convert categories to numeric codes

        # Create figure
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            x=self.results_df['Número de variáveis'],
            y=self.results_df['Menor Energia'],
            c=colors,  # Use numeric codes for color mapping
            cmap='viridis',  # Choosing a color map
            marker='o'  # Dot marker
        )

        # Set logarithmic scale for the y-axis
        plt.yscale('log')

        # Adding labels and title
        plt.xlabel('Número de variáveis')
        plt.ylabel('Menor energia (escala logarítimica)')
        plt.title('Menor energia obtida nos experimentos')

        # Custom discrete legend based on unique samplers
        unique_samplers = amostrador_categories.cat.categories
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(i / len(unique_samplers)), markersize=10) 
            for i, sampler in enumerate(unique_samplers)
        ]
        plt.legend(handles, unique_samplers, title="Amostrador", bbox_to_anchor=(1.05, 1), loc='upper left')

        # Save the plot to the specified directory
        os.makedirs(self.results_dir_path, exist_ok=True)  # Ensure the directory exists
        output_path = os.path.join(self.results_dir_path, 'minimum_energies_vs_variables.png')
        plt.tight_layout()
        plt.savefig(output_path)

        # Close the plot to free memory
        plt.close()
