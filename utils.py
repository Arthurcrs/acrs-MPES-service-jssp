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