import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import datetime
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import json

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

def get_percentage_of_valid_results(energies, makespan_function_max_value):
    total_count = len(energies)
    valid_count = np.sum(energies <= makespan_function_max_value)
    valid_percentage = (valid_count / total_count) * 100
    return valid_percentage

def new_export_gantt_diagram(gantt_chart_path, image_title, solution_csv_file_path, machine_downtimes, timespan=None, title=None):
    df = pd.read_csv(solution_csv_file_path)
    unique_jobs = df['job'].unique()
    conditions = [(df['job'] == job) for job in unique_jobs]
    values = get_colors(len(unique_jobs))
    df['color'] = np.select(conditions, values)

    fig, ax = plt.subplots(figsize=(8, 8))
    axx = ax.barh(df['machine'], df['duration'], align='center', left=df['start_time'], 
                  color=df['color'], edgecolor='black', linewidth=1.5, label=df['position'], alpha=0.85, height=1.0)

    for machine, downtimes in machine_downtimes.items():
        for downtime in downtimes:
            ax.barh(machine, 1, left=downtime, color='none', edgecolor='black', 
                    linewidth=1.5, height=1.0, hatch='//', label='Downtime')

    max_x = timespan if timespan else df['end_time'].max()
    ax.set_xlim(0, max_x)
    all_machines = sorted(set(df['machine']).union(machine_downtimes.keys()))
    ax.set_yticks(all_machines)
    ax.set_ylim(min(all_machines) - 0.5, max(all_machines) + 0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(np.arange(0, max_x + 1, 1))
    ax.set_xlabel('Instante')
    ax.set_ylabel('Máquina', labelpad=10)

    handles = []
    for job, color in zip(pd.unique(df['job']), pd.unique(df['color'])):
        handles.append(mpatches.Patch(color=color, label=job))

    handles.append(mpatches.Patch(facecolor='none', edgecolor='black', hatch='//', label='Inatividade'))
    plt.legend(handles=handles, title='Trabalhos', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.bar_label(axx, df['position'], label_type='center')
    ax.grid(False)
    plt.title(title)
    plt.tight_layout(pad=0.5, rect=[0, 0, 0.85, 1])
    plt.savefig(gantt_chart_path + image_title + '.png', bbox_inches='tight')
    plt.close()

def isQuantumSampler(sampler_name):
    if sampler_name == 'DwaveSampler' or sampler_name == 'LeapHybridSampler':
        return True
    return False

def export_sjssp_as_text(sjssp, result_path):
    with open(result_path + 'sjssp_text.txt', 'w') as file:
        for key, value in sjssp.items():
            file.write(f'{key}: {format_value(value)}\n')

def format_value(value, indent=4):
    if isinstance(value, dict):
        formatted_items = [f"{k}: {format_value(v, indent + 4)}" for k, v in value.items()]
        return "{\n" + ",\n".join(" " * indent + item for item in formatted_items) + "\n" + " " * (indent - 4) + "}"
    elif isinstance(value, list):
        if all(not isinstance(i, (list, dict)) for i in value):
            return "[" + ", ".join(format_value(i) for i in value) + "]"
        else:
            return "[\n" + ",\n".join(" " * (indent + 4) + format_value(i, indent + 4) for i in value) + "\n" + " " * indent + "]"
    else:
        return str(value)