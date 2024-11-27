import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import *
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

number_of_variables = 150
results_dir_path = 'Simulation/Results/' + str(number_of_variables) + '_variables/Chain Strength Tests/'

results_df = pd.read_csv(results_dir_path + 'chain_strength_test.csv')

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(results_df['Chain Strength'], results_df['Menor energia'], marker='o', color='blue', label='Menor Energia')
ax1.set_xlabel('Força da cadeia', color='black', fontsize=14)
ax1.set_ylabel('Menor Energia', color='black', fontsize=14)
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(results_df['Chain Strength'], results_df['Percentual médio de chainbreaks entre as amostras'], 
         marker='^', color='red', label='Média de quebra de cadeias (%)')
ax2.set_ylabel('Média de quebra de cadeias (%)', color='black', fontsize=14)
ax2.tick_params(axis='y', labelcolor='red')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.savefig(results_dir_path + str(number_of_variables) + '_chain_breaks.png')
plt.close()
