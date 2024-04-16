import time
import numpy as np

class Time_Counter:
    def __init__(self):
        self.loop_counter: int = 0
        self.recorded_times = [] # Timetags at different points in code
        self.times_taken = [] # Difference of values in self.recorded_times = time taken to run each part
        self.times_integrated = [] # Integral of self.times_taken
    def add_time(self):
        self.recorded_times.append(time.time())
        # print(f'{self.recorded_times}')
        
        # Fix size of integrated
        self.times_integrated.append(0)
    def new_iter(self):
        '''
        Run at the end of each iteration
        
        Add self.recorded_times to self.times_integrated
        Calculate fps from the times
        '''
        # Set the size of the integrated array at first iteration
        if self.loop_counter == 0:
            self.times_integrated.append([0] * len(self.times_taken))
        
        self.times_taken = [(self.recorded_times[i+1] - self.recorded_times[i]) for i in range(len(self.recorded_times)-1)]
        self.times_integrated = [integ + new for integ, new in zip(self.times_integrated, self.times_taken)]
        self.loop_counter += 1
        
        # print(f'Times taken:')
        # for n in self.times_taken:
        #     print(f'{n:.3f}')
        
        # Reset counters for next iteration. times_integrated remains obviously.
        self.recorded_times = []
        self.times_taken = []
    def end_process(self):
        '''
        When the measuring process is finished

        '''
        self.loop_counter = 0
        self.recorded_times = []
        self.times_taken = []
        self.times_integrated = []