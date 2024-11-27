import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import pickle

from utils import *


class SimulationGraphsManager:

    def __init__(self, results_dir, output_path, number_of_variables):
        self.results_dir = results_dir
        self.output_path = output_path
        self.results_df = pd.read_csv(results_dir + 'results.csv')
        self.energies_df = pd.read_csv(results_dir + 'energies.csv')
        self.number_of_variables = number_of_variables
    
        self.sampler_colors = {
                'DwaveSampler': '#FF0000',  
                'LeapHybridSampler': '#0000FF',  
                'SimulatedAnnealing': '#006400',  
                'TabuSampler': '#8B4513',  
                'SteepestDescentSampler': '#FF00FF'  
            }
        self.label_font_size = 14
        self.column_width_factor = 0.5

    def plot_lowest_energies(self):
        samplers = self.results_df['Amostrador']
        lowest_energies = self.results_df['Menor energia']
        
        colors = [ self.sampler_colors.get(sampler, '#000000') for sampler in samplers]

        plt.figure(figsize=(10, 6))
        plt.bar(samplers, lowest_energies, color=colors, width=self.column_width_factor)
        plt.ylabel('Menor energia', fontsize=self.label_font_size )
        plt.xticks(rotation=45, fontsize=10)
        plt.yscale('log')
        plt.tight_layout()

        plt.savefig(self.output_path + str(self.number_of_variables) + '_lowest_energies.png')
        plt.close()

    def plot_percent_of_valid_solutions(self):
        filtered_data = self.results_df
        samplers = filtered_data['Amostrador']
        percent_valid = filtered_data['Percentual de soluções válidas']
        colors = [self.sampler_colors.get(sampler, '#000000') for sampler in samplers]

        plt.figure(figsize=(10, 6))
        plt.bar(samplers, percent_valid, color=colors, width=self.column_width_factor)
        plt.ylabel('Percentual de soluções válidas', fontsize=self.label_font_size)
        plt.xticks(rotation=45, fontsize=10)
        plt.tight_layout()

        plt.savefig(self.output_path + str(self.number_of_variables) + '_valid_solutions.png')
        plt.close()

    def plot_energies_histogram(self):
        plt.figure(figsize=(10, 6))
        for sampler, color in self.sampler_colors.items():
            sampler_energies = self.energies_df[self.energies_df['Amostrador'] == sampler]['Energia']
            plt.hist(sampler_energies, bins=100, color=color, label=sampler, log=True)

        plt.xlabel('Energia', fontsize=self.label_font_size)
        plt.ylabel('Frequência', fontsize=self.label_font_size)
        plt.title('Histograma de Energias')
        plt.xscale('log')
        plt.legend(title='Amostrador')
        plt.tight_layout()

        plt.savefig(self.output_path + str(self.number_of_variables) + '_energies_histogram.png')
        plt.close()

