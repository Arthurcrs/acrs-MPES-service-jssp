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
import seaborn as sns 

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

        self.sampler_colors = {
            'DwaveSampler': '#FF0000',  
            'LeapHybridSampler': '#0000FF',  
            'SimulatedAnnealing': '#006400',  
            'TabuSampler': '#8B4513',  
            'SteepestDescentSampler': '#FF00FF'  
        }

    def init_results_df(self):
        columns = [
            'Amostrador',
            'Número de trabalhos',
            'Número de máquinas distintas utilizadas',
            'Número de equipamentos distintos utilizados',
            'Número de operações',
            'Timespan',
            'Número de variáveis',
            'Número de interações',
            'Menor Energia',
            'Tempo de amostragem',
            'Percentual de soluções válidas',
            'Parâmetro: Número de trabalhos',
            'Parâmetro: Máximo número de operações em um trabalho',
            'Parâmetro: Número de máquinas',
            'Parâmetro: Número de equipamentos possíveis',
            'qpu_sampling_time',
            'qpu_readout_time_per_sample',
            'qpu_access_overhead_time',
            'qpu_anneal_time_per_sample',
            'qpu_access_time',
            'qpu_programming_time',
            'qpu_delay_time_per_sample',
            'total_post_processing_time',
            'post_processing_overhead_time',
            'Percentual médio de chainbreaks entre as amostras',
            'Percentual de amostras com alta quantidade de chain breaks (>15%)'
        ]
        self.results_df = pd.DataFrame(columns=columns)

    def set_results_df(self, results_df):
        self.results_df = results_df

    def set_results_path(self, results_dir_path):
        self.results_dir_path = results_dir_path

    def set_sampler_colors(self, sampler_colors):
        self.sampler_colors = sampler_colors

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
                        try:
                            start_time = time.time()
                            if sampler['name'] == 'LeapHybridSampler':
                                sampleset = sampler['sampler'].sample(bqm)
                            else:
                                sampleset = sampler['sampler'].sample(bqm,num_reads = self.n_reads)
                            if sampler['name'] == 'DwaveSampler':
                                qpu_timing_info = sampleset.info['timing']
                                chain_breaks_data = sampleset.record.chain_break_fraction
                                average_chain_break_percentage = np.mean(chain_breaks_data) * 100
                                percentage_of_samples_with_high_chain_breaks = 100 * np.count_nonzero(chain_breaks_data > 0.15)/len(chain_breaks_data)
                            else:
                                qpu_timing_info = None
                                average_chain_break_percentage = None
                                percentage_of_samples_with_high_chain_breaks = None
                            end_time = time.time()
                            sample_time = end_time - start_time
                            energies = sampleset.record.energy
                            self.add_results_to_df(sampler['name'], 
                                                sample_time, 
                                                jobs, timespan, 
                                                bqm, energies, 
                                                makespan_function_max_value, 
                                                parameters, 
                                                qpu_timing_info, 
                                                average_chain_break_percentage, 
                                                percentage_of_samples_with_high_chain_breaks)
                        except Exception as e:
                            print(f"An error occurred: {e}")
                pbar.update(1)
        experiment_end_time = time.time()
        print('Experiments finished in: ' + str(round((experiment_end_time - experiment_start_time)/60,1)) + ' minutes')

    def generate_sjssp(self, parameters):
        jobs = {}
        machine_downtimes = {}
        timespan = -1
        
        n_jobs = parameters['n_jobs']
        n_possible_machines = parameters['n_possible_machines']
        max_op_in_job = parameters['max_op_in_job']
        n_possible_equipments = parameters['n_possible_equipments']
        max_operation_duration = self.max_operation_duration

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
        timespan = math.ceil((min_timespan + max_timespan)/2)

        first_half_timespan = timespan // 2
        for machine_id in range(1, n_possible_machines + 1):
            downtime_instant = random.randint(0, first_half_timespan - 1)
            machine_downtimes[machine_id] = [downtime_instant]

        return jobs, machine_downtimes, timespan

    def add_results_to_df(self, sampler_name, sample_time, jobs, timespan, bqm, energies, makespan_function_max_value, parameters, qpu_timing_info, average_chain_break_percentage, percentage_of_samples_with_high_chain_breaks):
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

        if sampler_name == 'DwaveSampler':
            for key, value in qpu_timing_info.items():
                new_row[key] = value
            new_row['Percentual médio de chainbreaks entre as amostras'] = average_chain_break_percentage
            new_row['Percentual de amostras com alta quantidade de chain breaks (>15%)'] = percentage_of_samples_with_high_chain_breaks
        
        self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)

    def export_results_to_csv(self):
        os.makedirs(self.results_dir_path, exist_ok=True)
        self.results_df.to_csv(self.results_dir_path + 'results.csv', index=False)

    def plot_minimum_energies_vs_variables(self):
        
        plt.figure(figsize=(10, 6))
        for sampler, color in self.sampler_colors.items():
            sampler_data = self.results_df[self.results_df['Amostrador'] == sampler]
            plt.scatter(
                x=sampler_data['Número de variáveis'],
                y=sampler_data['Menor Energia'],
                color=color,
                label=sampler,
                marker='o',
                s=7.5
            )

        plt.yscale('log')
        plt.xlabel('Variáveis', fontsize=14)
        plt.ylabel('Menor energia', fontsize=14)

        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, linestyle='') 
            for sampler, color in self.sampler_colors.items()
        ]
        plt.legend(handles, self.sampler_colors.keys(), title="Amostrador", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
        plt.grid(True, alpha=0.3)
        os.makedirs(self.results_dir_path, exist_ok=True)
        output_path = os.path.join(self.results_dir_path, 'minimum_energies_vs_variables.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_sampling_time_vs_variables(self):
        plt.figure(figsize=(10, 6))

        for sampler, color in self.sampler_colors.items():
            sampler_data = self.results_df[self.results_df['Amostrador'] == sampler]
            plt.scatter(
                x=sampler_data['Número de variáveis'],
                y=sampler_data['Tempo de amostragem'],
                color=color,
                label=sampler,
                marker='o',
                s=7.5
            )

        plt.yscale('log')
        plt.xlabel('Variáveis', fontsize=14)
        plt.ylabel('Tempo de amostragem (segundos)', fontsize=14)

        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, linestyle='') 
            for sampler, color in self.sampler_colors.items()
        ]
        plt.legend(handles, self.sampler_colors.keys(), title="Amostrador", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
        os.makedirs(self.results_dir_path, exist_ok=True)
        output_path = os.path.join(self.results_dir_path, 'sampling_time_vs_variables.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_best_sampler_per_test(self):
        plt.figure(figsize=(10, 6))

        test_columns = [
            'Parâmetro: Número de trabalhos',
            'Parâmetro: Máximo número de operações em um trabalho',
            'Parâmetro: Número de máquinas',
            'Parâmetro: Número de equipamentos possíveis'
        ]

        best_results = self.results_df.loc[self.results_df.groupby(test_columns)['Menor Energia'].idxmin()]

        for sampler, color in self.sampler_colors.items():
            sampler_data = best_results[best_results['Amostrador'] == sampler]
            plt.scatter(
                x=sampler_data['Número de variáveis'],
                y=[sampler] * len(sampler_data),
                color=color,
                label=sampler,
                marker='o',
                s=7.5
            )

        plt.xlabel('Variáveis', fontsize=14)
        plt.ylabel('Amostrador', fontsize=14)
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, linestyle='') 
            for sampler, color in self.sampler_colors.items()
        ]
        plt.legend(handles, self.sampler_colors.keys(), title="Amostrador", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)

        os.makedirs(self.results_dir_path, exist_ok=True)
        output_path = os.path.join(self.results_dir_path, 'best_sampler_per_test.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_best_sampler_percentage_per_bin(self):
        self.results_df['Variable_Bin'] = (self.results_df['Número de variáveis'] // 25) * 25

        test_columns = [
            'Parâmetro: Número de trabalhos',
            'Parâmetro: Máximo número de operações em um trabalho',
            'Parâmetro: Número de máquinas',
            'Parâmetro: Número de equipamentos possíveis'
        ]

        ranked_results = (
            self.results_df
            .sort_values(by=['Menor Energia'])
            .groupby(test_columns + ['Variable_Bin'])
            .apply(lambda x: x.assign(Rank=range(1, len(x) + 1)))
            .reset_index(drop=True)
        )

        best_samplers = ranked_results[ranked_results['Rank'] == 1]
        best_sampler_counts = best_samplers.groupby(['Variable_Bin', 'Amostrador']).size().unstack(fill_value=0)
        best_sampler_percentages = (best_sampler_counts.T / best_sampler_counts.sum(axis=1)).T * 100
        bin_centers = best_sampler_percentages.index + 12.5

        plt.figure(figsize=(10, 8))
        for sampler, color in self.sampler_colors.items():
            if sampler in best_sampler_percentages.columns:
                plt.plot(
                    bin_centers,  
                    best_sampler_percentages[sampler],
                    color=color,
                    label=sampler,
                    marker='o',
                    markersize=7.5
                )

        plt.xlabel('Variáveis', fontsize=14)
        plt.ylabel('Porcentagem de vezes em que o melhor resultado foi obtido', fontsize=14)
        plt.xticks(bin_centers, [f"{int(bin)}-{int(bin + 24)}" for bin in best_sampler_percentages.index], rotation=45)
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, linestyle='') 
            for sampler, color in self.sampler_colors.items()
        ]
        plt.legend(handles, self.sampler_colors.keys(), title="Amostrador", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
        plt.grid(True, alpha=0.3)
        os.makedirs(self.results_dir_path, exist_ok=True)
        output_path = os.path.join(self.results_dir_path, 'best_sampler_percentage_per_bin.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_average_chain_breaks_vs_variables(self):
        dwave_data = self.results_df[self.results_df['Amostrador'] == 'DwaveSampler']

        plt.figure(figsize=(10, 6))
        plt.scatter(
            x=dwave_data['Número de variáveis'],
            y=dwave_data['Percentual médio de chainbreaks entre as amostras'],
            color='blue',
            marker='o',
            s=25,
        )

        plt.xlabel('Variáveis', fontsize=14)
        plt.ylabel('Percentual médio de chain breaks', fontsize=14)
        plt.grid(True, alpha=0.3)
        os.makedirs(self.results_dir_path, exist_ok=True)
        output_path = os.path.join(self.results_dir_path, 'average_chain_breaks_vs_variables.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_variables_vs_interactions(self):
        dwave_data = self.results_df[self.results_df['Amostrador'] == 'DwaveSampler']

        plt.figure(figsize=(10, 6))
        plt.scatter(
            x=dwave_data['Número de variáveis'],
            y=dwave_data['Número de interações'],
            color='blue',
            marker='o',
            s=25,
        )

        plt.xlabel('Variáveis', fontsize=14)
        plt.ylabel('Interações', fontsize=14)
        plt.grid(True, alpha=0.3)

        os.makedirs(self.results_dir_path, exist_ok=True)
        output_path = os.path.join(self.results_dir_path, 'variables_vs_interactions.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_sampler_ranks_per_bin(self):
        self.results_df['Variable_Bin'] = (self.results_df['Número de variáveis'] // 25) * 25

        test_columns = [
            'Parâmetro: Número de trabalhos',
            'Parâmetro: Máximo número de operações em um trabalho',
            'Parâmetro: Número de máquinas',
            'Parâmetro: Número de equipamentos possíveis'
        ]

        ranked_results = (
            self.results_df
            .sort_values(by=['Menor Energia'])
            .groupby(test_columns + ['Variable_Bin'])
            .apply(lambda x: x.assign(Rank=range(1, len(x) + 1)))
            .reset_index(drop=True)
        )

        best_sampler_counts = ranked_results[ranked_results['Rank'] == 1].groupby(['Variable_Bin', 'Amostrador']).size().unstack(fill_value=0)
        best_sampler_ranks = best_sampler_counts.rank(axis=1, method='min', ascending=False)
        bin_centers = best_sampler_ranks.index + 12.5

        plt.figure(figsize=(10, 8))

        for sampler, color in self.sampler_colors.items():
            if sampler in best_sampler_ranks.columns:
                plt.plot(
                    bin_centers,
                    best_sampler_ranks[sampler],
                    color=color,
                    label=sampler,
                    marker='o',
                    markersize=7.5
                )

        plt.xlabel('Variáveis', fontsize=14)
        plt.ylabel('Rank (1 = Melhor)', fontsize=14)
        max_rank = best_sampler_ranks.shape[1]
        plt.yticks(range(1, max_rank + 1))
        plt.gca().invert_yaxis()
        plt.xticks(bin_centers, [f"{int(bin)}-{int(bin + 24)}" for bin in best_sampler_ranks.index], rotation=45)

        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, linestyle='') 
            for sampler, color in self.sampler_colors.items()
        ]
        plt.legend(handles, self.sampler_colors.keys(), title="Amostrador", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
        plt.grid(True, alpha=0.3)
        os.makedirs(self.results_dir_path, exist_ok=True)
        output_path = os.path.join(self.results_dir_path, 'sampler_ranks_per_bin.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_best_sampler_percentage_per_bin_closer(self):
        filtered_df = self.results_df[self.results_df['Número de variáveis'] <= 75]

        filtered_df['Variable_Bin'] = (filtered_df['Número de variáveis'] // 15) * 15

        test_columns = [
            'Parâmetro: Número de trabalhos',
            'Parâmetro: Máximo número de operações em um trabalho',
            'Parâmetro: Número de máquinas',
            'Parâmetro: Número de equipamentos possíveis'
        ]

        ranked_results = (
            filtered_df
            .sort_values(by=['Menor Energia']) 
            .groupby(test_columns + ['Variable_Bin'])
            .apply(lambda x: x.assign(Rank=range(1, len(x) + 1))) 
            .reset_index(drop=True)
        )

        best_samplers = ranked_results[ranked_results['Rank'] == 1]
        best_sampler_counts = best_samplers.groupby(['Variable_Bin', 'Amostrador']).size().unstack(fill_value=0)
        best_sampler_percentages = (best_sampler_counts.T / best_sampler_counts.sum(axis=1)).T * 100
        bin_centers = best_sampler_percentages.index + 7.5
        plt.figure(figsize=(10, 8))
        for sampler, color in self.sampler_colors.items():
            if sampler in best_sampler_percentages.columns:
                plt.plot(
                    bin_centers,
                    best_sampler_percentages[sampler],
                    color=color,
                    label=sampler,
                    marker='o',
                    markersize=7.5
                )

        plt.xlabel('Variáveis', fontsize=14)
        plt.ylabel('Porcentagem de vezes em que o melhor resultado foi obtido', fontsize=14)
        plt.title('Porcentagem de Melhor Resultado por Bin de Variáveis (até 75 variáveis)', fontsize=14)
        plt.xticks(bin_centers, [f"{int(bin)}-{int(bin + 14)}" for bin in best_sampler_percentages.index], rotation=45)
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, linestyle='') 
            for sampler, color in self.sampler_colors.items()
        ]
        plt.legend(handles, self.sampler_colors.keys(), title="Amostrador", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)

        plt.grid(True, alpha=0.3)
        os.makedirs(self.results_dir_path, exist_ok=True)
        output_path = os.path.join(self.results_dir_path, 'best_sampler_percentage_per_bin_closer.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_high_chain_break_percentage_vs_variables(self):
        dwave_data = self.results_df[self.results_df['Amostrador'] == 'DwaveSampler']

        plt.figure(figsize=(10, 6))
        plt.scatter(
            x=dwave_data['Número de variáveis'],
            y=dwave_data['Percentual de amostras com alta quantidade de chain breaks (>15%)'],
            color='green',
            marker='o',
            s=25,
        )

        plt.xlabel('Variáveis', fontsize=14)
        plt.ylabel('Percentual de amostras com alta quantidade de chain breaks (>15%)', fontsize=10)
        plt.grid(True, alpha=0.3)
        os.makedirs(self.results_dir_path, exist_ok=True)
        output_path = os.path.join(self.results_dir_path, 'high_chain_break_percentage_vs_variables.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_operations_vs_machines(self):
        data = self.results_df

        plt.figure(figsize=(10, 6))
        plt.scatter(
            x=data['Número de operações'],
            y=data['Número de máquinas distintas utilizadas'],
            color='purple',
            marker='o',
            s=25,
        )

        plt.xlabel('Número de Operações', fontsize=14)
        plt.ylabel('Número de Máquinas', fontsize=14)
        plt.grid(True, alpha=0.3)
        os.makedirs(self.results_dir_path, exist_ok=True)
        output_path = os.path.join(self.results_dir_path, 'operations_vs_machines.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_operations_vs_machines_heatmap(self):
        heatmap_data_trim = self.results_df[self.results_df['Amostrador'] == 'DwaveSampler']
        heatmap_data = heatmap_data_trim.groupby(
            ['Número de máquinas distintas utilizadas', 'Número de operações']
        ).size().unstack(fill_value=0)

        plt.figure(figsize=(10, 6))
        sns.heatmap(
            heatmap_data,
            annot=True,
            cmap="YlGnBu",
            cbar_kws={'label': 'Quantidade de Ocorrências'}
        )

        plt.gca().invert_yaxis()
        plt.xlabel('Número de Operações', fontsize=14)
        plt.ylabel('Número de Máquinas', fontsize=14)
        os.makedirs(self.results_dir_path, exist_ok=True)
        output_path = os.path.join(self.results_dir_path, 'operations_vs_machines_heatmap.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
