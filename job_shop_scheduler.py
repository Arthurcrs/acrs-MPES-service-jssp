from utils import *
from dimod import BinaryQuadraticModel

def get_jss_bqm(job_dict, machine_downtimes, timespan=None):
    scheduler = JobShopScheduler(job_dict, machine_downtimes, timespan)
    return scheduler.get_bqm()

def get_label(task, machine, time):
    return f"{task.job}_{task.position},{machine},{time}".format(**locals())

class Task:
    def __init__(self, job, position, machines, equipments, duration):
        self.job = job
        self.position = position
        self.machines = machines
        self.equipments = equipments
        self.duration = duration

class JobShopScheduler:
    def __init__(self, job_dict, machine_downtimes, timespan=None):
        self.bqm = BinaryQuadraticModel.empty('BINARY')
        self.machine_downtimes = machine_downtimes
        self.removed_labels = []
        self.tasks = []
        self.last_task_indices = []
        self.timespan = timespan
        self.machines = []
        self.equipments = []
        self.tasks_with_machine = {}
        self.tasks_with_equipment = {}
        self.jobs_min_end_times = []
        self.number_of_jobs = 0
        self.number_of_machines = 0

        self.one_start_constraint_penalty = 1
        self.precedence_constraint_penalty = 1
        self.share_machine_constraint_penalty = 1
        self.share_equipment_constraint_penalty = 1
        self.makespan_function_max_value = 1

        self._process_data(job_dict)
    
    def _process_data(self, jobs):
        tasks = []
        last_task_indices = [-1]
        total_time = 0

        # Populates self.tasks, self.timespan, self.machines and self.equipments
        for job_name, job_tasks in jobs.items():
            last_task_indices.append(last_task_indices[-1] + len(job_tasks))
            job_time = 0

            for i, (machines, equipments, time_span) in enumerate(job_tasks):
                tasks.append(Task(job_name, i, machines, equipments, time_span))
                total_time += time_span
                job_time += time_span

                for machine in machines:
                    if machine not in self.machines:
                        self.machines.append(machine)

                for equipment in equipments:
                    if equipment not in self.equipments:
                        self.equipments.append(equipment)

            self.jobs_min_end_times.append(job_time)

        self.tasks = tasks
        self.last_task_indices = last_task_indices[1:]
        self.number_of_jobs = len(self.last_task_indices)
        self.number_of_machines = len(self.machines)

        if self.timespan is None:
            self.timespan = total_time
        self.timespan -= 1

        # Populates self.tasks_with_machine
        for m in self.machines:
            for task in self.tasks:
                if m in task.machines:
                    if m not in self.tasks_with_machine:
                        self.tasks_with_machine[m] = []
                    self.tasks_with_machine[m].append(task)

        # Populates self.tasks_with_equipment
        for e in self.equipments:
            for task in self.tasks:
                if e in task.equipments:
                    if e not in self.tasks_with_equipment:
                        self.tasks_with_equipment[e] = []
                    self.tasks_with_equipment[e].append(task)
        
        # Constraint penalty calculation 
        self.makespan_function_max_value = self.number_of_jobs * self.number_of_machines * (self.number_of_jobs + 1)**(self.timespan) # maximum possible value of the timespan function
        self.one_start_constraint_penalty = self.makespan_function_max_value * 2
        self.precedence_constraint_penalty = self.makespan_function_max_value * 2
        self.share_machine_constraint_penalty = self.makespan_function_max_value * 2
        self.share_equipment_constraint_penalty = self.makespan_function_max_value * 2
   
    def add_makespan_function(self):
        """
            Condition that penalizes solutions that the job is not finished as soon as possible
        """
        for i, job_min_end_time in zip(self.last_task_indices, self.jobs_min_end_times):
            last_task = self.tasks[i]
            for t in range(job_min_end_time - last_task.duration + 1, self.timespan + 1):
                job_end_time = t + last_task.duration
                bias = (self.number_of_jobs + 1)**(job_end_time - job_min_end_time)
                for m in last_task.machines:
                    label = get_label(last_task,m,t)
                    pruned_variables = list(self.bqm.variables)
                    if label in pruned_variables:
                        self.bqm.add_variable(label, bias)

    def add_one_start_constraint(self):
        """
            A task can start once and only once
        """
        for task in self.tasks:
            linear_terms = []
            for m in task.machines:
                for t in range(self.timespan + 1):
                    label = get_label(task, m, t)
                    linear_terms.append((label, 1.0))
            self.bqm.add_linear_equality_constraint(linear_terms, self.one_start_constraint_penalty, -1)

    def add_precedence_constraint(self):
        """
            The tasks within a job must be executed in order
        """
        for current_task, next_task in zip(self.tasks, self.tasks[1:]):
            if current_task.job != next_task.job:
                continue
            for t in range(self.timespan + 1):
                for tt in range(min(t + current_task.duration, self.timespan + 1)):
                    current_labels = []
                    next_labels = []
                    quadratic_terms = []
                    for m in current_task.machines:
                        current_labels.append(get_label(current_task, m, t))
                    for m in next_task.machines:
                        next_labels.append(get_label(next_task, m, tt))                   
                    quadratic_terms = [(u, v) for u in current_labels for v in next_labels]
                    for current_label, next_label in quadratic_terms:
                        self.bqm.add_quadratic(current_label, next_label, self.precedence_constraint_penalty)
    
    def add_share_machine_constraint(self):
        """
            There can be only one job running on each machine at any given point in time
        """  
        for m in self.machines:
            task_pairs = [(task_1, task_2) for task_1 in self.tasks_with_machine[m] for task_2 in self.tasks_with_machine[m]]
            for task_1, task_2 in task_pairs:
                if task_1.job != task_2.job or task_1.position != task_2.position:
                    for t_1 in range(self.timespan + 1):
                        for t_2 in range(t_1, min(t_1 + task_1.duration, self.timespan + 1)):
                            label_1 = get_label(task_1, m, t_1)
                            label_2 = get_label(task_2, m, t_2)
                            self.bqm.add_quadratic(label_1,label_2, self.share_machine_constraint_penalty)                     

    def add_share_equipment_constraint(self):
        """
            There can be only one task using a equipment at any given point in time
        """  
        for task_1 in self.tasks:
            for m_1 in task_1.machines:
                for t_1 in range(self.timespan + 1):
                    for e in task_1.equipments:
                        for task_2 in self.tasks_with_equipment[e]:
                            if task_1.job != task_2.job or task_1.position != task_2.position:
                                for t_2 in range(t_1, min(t_1 + task_1.duration, self.timespan + 1)):
                                    for m_2 in task_2.machines:
                                        label_1 = get_label(task_1, m_1, t_1)
                                        label_2 = get_label(task_2, m_2, t_2)
                                        self.bqm.add_quadratic(label_1,label_2, self.share_equipment_constraint_penalty)

    def _remove_machine_downtime_labels(self):
        """
            Sets labels with impossible task times due to machine downtimes to 0.
        """
        for m in self.machine_downtimes.keys():
            if m in self.tasks_with_machine:
                for task in self.tasks_with_machine[m]:
                    for begin_downtime in self.machine_downtimes[m]:
                        start = begin_downtime - task.duration + 1
                        stop = begin_downtime
                        if start < 0:
                            start = 0
                        for time in range(start,stop + 1):
                            label = get_label(task, m, time)
                            if label not in self.removed_labels:
                                self.bqm.fix_variable(label, 0)
                                self.removed_labels.append(label)
        
    def _remove_absurd_times_labels(self):
        """
            Sets labels with task times that are too early or too late for a task to be executed to 0.
        """
        # Times that are too early for task
        predecessor_time = 0
        current_job = self.tasks[0].job
        for task in self.tasks:
            if task.job != current_job:
                predecessor_time = 0
                current_job = task.job

            for t in range(predecessor_time):
                if t > self.timespan:
                    continue
                for m in task.machines:
                    label = get_label(task, m, t)
                    if label not in self.removed_labels:
                        self.bqm.fix_variable(label, 0)
                        self.removed_labels.append(label)

            predecessor_time += task.duration

        # Times that are too late for task
        successor_time = -1
        current_job = self.tasks[-1].job
        for task in self.tasks[::-1]:
            if task.job != current_job:
                successor_time = -1
                current_job = task.job

            successor_time += task.duration
            for t in range(successor_time):
                if self.timespan - t < 0:
                    continue
                for m in task.machines:
                    label = get_label(task, m, self.timespan - t)
                    if label not in self.removed_labels:
                        self.bqm.fix_variable(label, 0)
                        self.removed_labels.append(label)
    
    def get_bqm(self):  

        # Adding Constraints
        self.add_one_start_constraint()
        self.add_share_machine_constraint()
        self.add_precedence_constraint()
        self.add_share_equipment_constraint()

        # Add timespan function
        self.add_makespan_function()

        # Pruning Variables
        self._remove_absurd_times_labels()
        self._remove_machine_downtime_labels()

        return self.bqm