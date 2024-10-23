import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import datetime
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

def parse_label(label):
    job_position, machine, start_time = label.split(',')
    job  = job_position.split('_')[1]
    position = job_position.split('_')[2]
    return {
        'job': int(job),
        'position': int(position),
        'machine': int(machine),
        'start_time': int(start_time)
    }

def solution_to_dataframe(solution, jobs):
    job_names = []
    positions = []
    machines = []
    durations = []
    start_times = []

    for label, value in solution.items():
        if value == 1:
            parsed_label = parse_label(label)
            # Retrieve corresponding machine and duration from the problem definition
            duration = jobs[f"job_{parsed_label['job']}"][parsed_label['position']][2]

            # Append information to the lists
            job_names.append(parsed_label['job'])
            positions.append(parsed_label['position'])
            machines.append(parsed_label['machine'])
            durations.append(duration)
            start_times.append(parsed_label['start_time'])

    # Create a DataFrame from the extracted information
    df = pd.DataFrame({
        'job': job_names,
        'position': positions,
        'machine': machines,
        'duration': durations,
        'start_time': start_times
    })

    df['end_time'] = df['start_time'] + df['duration']

    return df

def get_colors(n):
    colormap = cm.get_cmap('tab20')
    colors = [mcolors.rgb2hex(colormap(i % 20 / 20)) for i in range(n)]
    return colors

def export_gantt_diagram(gantt_chart_path, image_title, solution_csv_file_path):

    df = pd.read_csv(solution_csv_file_path)
    unique_jobs = df['job'].unique()
    conditions = [(df['job'] == job) for job in unique_jobs]
    values = get_colors(len(unique_jobs))
    df['color'] = np.select(conditions, values)

    fig, ax = plt.subplots(figsize=(16,6))
    axx = ax.barh(df['machine'], df['duration'], align='center', left=df['start_time'], color=df['color'], label=df['position'], linewidth=3, alpha=.5)

    ax.set_xlim(0, df['end_time'].max())

    fig.text(0.5, 0.04, 'Time Unit', ha='center')
    fig.text(0.1, 0.5, 'machine', va='center', rotation='vertical')

    handles = []
    for job,color in zip(pd.unique(df['job']),pd.unique(df['color'])):
        handles.append(Patch(color=color, label=job))

    plt.legend(handles=handles, title='job')

    ax.set_yticks(df['machine'])
    ax.bar_label(axx, df['position'], label_type='center')

    plt.grid(axis = 'x')
    plt.savefig( gantt_chart_path + image_title + '.png')

    plt.close()

def count_unique_machines(jobs):
    machines_set = set()
    for job_key, operations in jobs.items():
        for operation in operations:
            machines, _, _ = operation
            machines_set.update(machines)
    return len(machines_set)

def count_unique_equipment(jobs):
    equipment_set = set()
    for job_key, operations in jobs.items():
        for operation in operations:
            _, equipment, _ = operation
            equipment_set.update(equipment)
    return len(equipment_set)

def count_total_operations(jobs):
    total_operations = 0
    for job_key, operations in jobs.items():
        total_operations += len(operations)
    return total_operations

"""
# This will generate a sample without machine downtimes, with a given number of jobs and timespan, all having the same number of operations, all operations
# can be executed by a set of a given number of machines, using a number of equipments, and with a set duration
"""
def generate_jssp_dict(n_jobs, n_operations_per_job, n_machines_per_operation, timespan, n_equipments_per_operation, operation_duration):
    jssp_dict = {
        "jobs": {},
        "machine_downtimes" : {},
        "timespan": timespan
    }
    
    for job_index in range(1, n_jobs + 1):
        job_key = f"job_{job_index}"
        operations = []
        
        for _ in range(n_operations_per_job):
            machines = list(range(1, n_machines_per_operation + 1))
            equipments = list(range(1, n_equipments_per_operation + 1))
            time = operation_duration
            operations.append((machines, equipments, time))
        
        jssp_dict["jobs"][job_key] = operations

    return jssp_dict

def generate_jssp_dict_based_on_size(n):
    jobs = {}
    for i in range(1, n+1):
        job_operations = []
        for j in range(1, n+1):
            machines = list(range(1, n+1))
            equipment = [j]
            operation = (machines, equipment, 1)
            job_operations.append(operation)
        jobs[f"job_{i}"] = job_operations

    timespan = n * n

    test_case = {
        "jobs": jobs,
        "machine_downtimes": {},
        "timespan": timespan
    }

    return test_case

def calculate_total_variables(jobs, timespan):
    total_variables = 0
    
    for job, operations in jobs.items():
        for operation in operations:
            machines_for_operation = len(operation[0])
            total_variables += machines_for_operation * timespan

    return total_variables

def get_current_datetime_as_string():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")

def new_export_gantt_diagram(gantt_chart_path, image_title, solution_csv_file_path, machine_downtimes, timespan=None):
    df = pd.read_csv(solution_csv_file_path)
    unique_jobs = df['job'].unique()
    conditions = [(df['job'] == job) for job in unique_jobs]
    values = get_colors(len(unique_jobs))
    df['color'] = np.select(conditions, values)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))  # Adjust figsize to get square proportions

    # Add edge color to bars to separate consecutive operations
    axx = ax.barh(df['machine'], df['duration'], align='center', left=df['start_time'], 
                  color=df['color'], edgecolor='black', linewidth=1.5, label=df['position'], alpha=0.85, height=1.0)  # Set height=1.0 to remove vertical spacing

    # Add machine downtimes with hatch patterns
    for machine, downtimes in machine_downtimes.items():
        for downtime in downtimes:
            ax.barh(machine, 1, left=downtime, color='none', edgecolor='black', 
                    linewidth=1.5, height=1.0, hatch='//', label='Downtime')

    # Set the x-axis limit based on timespan if provided, else use maximum end_time from data
    max_x = timespan if timespan else df['end_time'].max()
    ax.set_xlim(0, max_x)

    # Set y-axis to display all machines, even if they have no operations
    all_machines = sorted(set(df['machine']).union(machine_downtimes.keys()))  # Combine machines with jobs and downtimes
    ax.set_yticks(all_machines)  # Ensure all machines are included in y-axis

    # Set the limits for the y-axis to control scaling
    ax.set_ylim(min(all_machines) - 0.5, max(all_machines) + 0.5)  # Adjust to give some padding

    # Set equal scaling for both axes
    ax.set_aspect('equal', adjustable='box')

    # Force the x-axis to display all integer values
    ax.set_xticks(np.arange(0, max_x + 1, 1))  # Set x-ticks for every integer from 0 to max_x

    # Set x-axis and y-axis labels
    ax.set_xlabel('Instante')
    ax.set_ylabel('Máquina', labelpad=10)

    # Add legend for jobs and downtimes
    handles = []
    for job, color in zip(pd.unique(df['job']), pd.unique(df['color'])):
        handles.append(mpatches.Patch(color=color, label=job))

    # Add hatch pattern to legend
    handles.append(mpatches.Patch(facecolor='none', edgecolor='black', hatch='//', label='Inatividade'))

    # Move legend outside of the plot
    plt.legend(handles=handles, title='Trabalhos', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Set the y-ticks to represent the machines
    ax.bar_label(axx, df['position'], label_type='center')

    # Remove all gridlines
    ax.grid(False)

    # Adjust the layout to make room for the legend
    plt.tight_layout(pad=0.5, rect=[0, 0, 0.85, 1])  # Adjust layout to fit the legend outside the plot

    # Save the figure, using bbox_inches='tight' to remove extra white space
    plt.savefig(gantt_chart_path + image_title + '.png', bbox_inches='tight')
    plt.close()