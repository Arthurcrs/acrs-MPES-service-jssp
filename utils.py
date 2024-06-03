import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

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
    colormap = cm.viridis
    colors = [mcolors.rgb2hex(colormap(i/n)) for i in range(n)]
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
