from __future__ import print_function

from re import match
from bisect import bisect_right

from utils import *

from dimod import BinaryQuadraticModel

constraint_penalty = 10

def get_jss_bqm(job_dict, makespan=None, stitch_kwargs=None):
    if stitch_kwargs == None:
        stitch_kwargs = {}

    scheduler = JobShopScheduler(job_dict, makespan)
    return scheduler.get_bqm(stitch_kwargs)

class Task:
    def __init__(self, job, position, machines, duration):
        self.job = job
        self.position = position
        self.machines = machines
        self.duration = duration

    def __repr__(self):
        return ("{{job: {job}, position: {position}, machine: {machine}, duration:"
                " {duration}}}").format(**vars(self))

class KeyList:
    def __init__(self, array, key_function):
        self.array = array
        self.key_function = key_function

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        item = self.array[index]
        key = self.key_function(item)
        return key

def get_label(task, machine, time):
    return f"{task.job}_{task.position}_{machine},{time}".format(**locals())

def get_jss_bqm(job_dict, makespan=None, stitch_kwargs=None):
    if stitch_kwargs == None:
        stitch_kwargs = {}

    scheduler = JobShopScheduler(job_dict, makespan)
    return scheduler.get_bqm(stitch_kwargs)

class JobShopScheduler:
    def __init__(self, job_dict, makespan=None):
        self.tasks = []
        self.last_task_indices = []
        self.binary_vars = []
        self.labels = []
        self.makespan = makespan
        self.bqm = BinaryQuadraticModel.empty('BINARY')

        # Populates self.tasks and self.makespan
        self._process_data(job_dict)
    
    def _process_data(self, jobs):
        tasks = []
        last_task_indices = [-1]
        total_time = 0
        max_job_time = 0

        for job_name, job_tasks in jobs.items():
            last_task_indices.append(last_task_indices[-1] + len(job_tasks))
            job_time = 0

            for i, (machine, time_span) in enumerate(job_tasks):
                tasks.append(Task(job_name, i, machine, time_span))
                total_time += time_span
                job_time += time_span

            if job_time > max_job_time:
                max_job_time = job_time

        self.tasks = tasks
        self.last_task_indices = last_task_indices[1:]
        self.max_job_time = max_job_time - 1

        if self.makespan is None:
            self.makespan = total_time
        self.makespan -= 1
 

    def add_one_start_constraint(self):
        """
              A task can start once and only once
        """
        for task in self.tasks:
            linear_terms = []
            for m in task.machines:
                for t in range(self.makespan + 1):
                    label = get_label(task, m, t)
                    if(label not in self.labels):
                        linear_terms.append((label, 1.0))
            self.binary_vars.append(linear_terms)
            self.bqm.add_linear_equality_constraint(linear_terms, constraint_penalty, -1)
    
    def add_precedence_constraint(self):
        """
              self.modelcsp gets the constraint: Task must follow a particular order.
              Note: assumes self.tasks are sorted by jobs and then by position
        """
        for current_task, next_task in zip(self.tasks, self.tasks[1:]):
            if current_task.job != next_task.job:
                continue
            # Forming constraints with the relevant times of the next task
            for t in range(self.makespan + 1):
                current_labels = []
                for m in current_task.machines:
                    current_labels.append(get_label(current_task, m, t))
                for tt in range(min(t + current_task.duration, self.makespan + 1)):
                    next_labels = []
                    for m in next_task.machines:
                        next_labels.append(get_label(next_task, m, tt))
                    
                    quadratic_terms = [(u, v) for u in current_labels for v in next_labels]
                    for current_label,next_label in quadratic_terms:
                        if((current_label not in self.labels) and (next_label not in self.labels)):
                            self.bqm.add_quadratic(current_label, next_label, constraint_penalty)
    
    def add_share_machine_constraint(self):
        """
            self.modelcsp gets the constraint: At most one task per machine per time
        """
        sorted_tasks = sorted(self.tasks, key=lambda x: x.machine)
        wrapped_tasks = KeyList(sorted_tasks, lambda x: x.machine) # Key wrapper for bisect function
        head = 0
        while head < len(sorted_tasks):

            # Find tasks that share a machine
            tail = bisect_right(wrapped_tasks, sorted_tasks[head].machine)
            same_machine_tasks = sorted_tasks[head:tail]

            # Update
            head = tail

            # No need to build coupling for a single task
            if len(same_machine_tasks) < 2:
                continue

            # Apply constraint between all tasks for each unit of time
            for task in same_machine_tasks:
                for other_task in same_machine_tasks:
                    if task.job == other_task.job and task.position == other_task.position:
                        continue
                    for t in range(self.makespan + 1):
                        current_label = get_label(task, t)
                        for tt in range(t, min(t + task.duration, self.makespan + 1)):
                            other_label = get_label(other_task, tt)
                            if((current_label not in self.labels) and (other_label not in self.labels)):
                                self.bqm.add_quadratic(current_label, other_label, constraint_penalty)
    
    def _remove_absurd_times(self):
        """Sets impossible task times in self.bqm to 0.
        """
        # Times that are too early for task
        predecessor_time = 0
        current_job = self.tasks[0].job
        for task in self.tasks:
            # Check if task is in current_job
            if task.job != current_job:
                predecessor_time = 0
                current_job = task.job

            for t in range(predecessor_time):
                label = get_label(task, t)
                self.bqm.fix_variable(label, 0)

            predecessor_time += task.duration

        # Times that are too late for task
        # Note: we are going through the task list backwards in order to compute
        # the successor time
        successor_time = -1    # start with -1 so that we get (total task time - 1)
        current_job = self.tasks[-1].job
        for task in self.tasks[::-1]:
            # Check if task is in current_job
            if task.job != current_job:
                successor_time = -1
                current_job = task.job

            successor_time += task.duration
            for t in range(successor_time):
                label = get_label(task, self.makespan - t)
                self.bqm.fix_variable(label, 0)
    
    def get_bqm(self, stitch_kwargs=None):
        if stitch_kwargs is None:
            stitch_kwargs = {}
        

        # Apply constraints to self.model
        self.add_one_start_constraint()
        # self._remove_absurd_times()
        self.add_precedence_constraint()
        # self.add_share_machine_constraint()

        return self.bqm