import pandas as pd
import re

def merge_intervals(intervals):
    if not intervals:
        return []

    # Sort intervals based on start times
    sorted_intervals = sorted(intervals, key=lambda x: x[0])

    # Initialize the result list with the first interval
    result = [sorted_intervals[0]]

    # Merge overlapping intervals
    for interval in sorted_intervals[1:]:
        prev_interval = result[-1]
        if interval[0] <= prev_interval[1]:  # Overlapping intervals
            prev_interval[1] = max(prev_interval[1], interval[1])
        else:
            result.append(interval)

    return result

def add_label_to_remove(my_list, string_to_add):
    if string_to_add not in my_list:
        my_list.append(string_to_add)
    return my_list

def solution_to_dataframe(solution, jobs):
    # Initialize lists to store job, position (task), machine, duration, and start_time information
    job_names = []
    positions = []
    machines = []
    durations = []
    start_times = []

    # Iterate through the solution and extract relevant information
    for key, value in solution.items():
        if key.startswith('job') and value == 1:
            # Splitting the key to extract job, task (position), and start time
            job_task_time = key.split('_')
            job_name = '_'.join(job_task_time[:2])
            task, start_time = job_task_time[-1].split(',')

            # Convert task and start_time to integers
            task = int(task)
            start_time = int(start_time)

            # Retrieve corresponding machine and duration from the problem definition
            machine, duration = jobs[job_name][task]

            # Append information to the lists
            job_names.append(get_numeric_part(job_name))
            positions.append(task)
            machines.append(get_numeric_part(machine))
            durations.append(duration)
            start_times.append(start_time)

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

def get_numeric_part(s):
    result = re.findall(r'\d+', s)
    return int(''.join(result)) if result else None