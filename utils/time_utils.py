'''
Author: Ayush Goel (aygoel@seas.upenn.edu)
'''
from abc import ABC, abstractmethod 
import time
import torch


class Timer(ABC):

    def __init__(self, timelogs):
        self.timelogs = timelogs

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self):
        pass


class CPUTimer(Timer):
    
    def __enter__(self):
        self.start = time.time()

    def __exit__(self):
        end = time.time()
        self.timelogs.append( ( end - self.start ) * 1000 )


class GPUTimer(Timer):
        
    def __enter__(self):
        self.start_timer = torch.cuda.Event(enable_timing=True)
        self.end_timer = torch.cuda.Event(enable_timing=True)
        self.start_event.record()

    def __exit__(self):
        self.end.record()
        torch.cuda.synchronize()
        self.timelogs.append(self.start.elapsed_time(self.end))