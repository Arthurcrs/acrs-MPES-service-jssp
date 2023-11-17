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

def export_gantt_diagram(image_title, solution_csv_file_path):
    directory_path = "Results/"

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
    plt.savefig( directory_path + image_title + '.png')