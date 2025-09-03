import time
from contextlib import contextmanager
import torch
 
# Dummy TimeLogger for when no timing is requested
class DummyTimeLogger:
    @contextmanager
    def time_block(self):
        yield
        
class TimeLogger:
    def __init__(self):
        self.durations = []
        self.active_timing = False
        self.start_event = None
        self.end_event = None
        self.reset()
    
    def reset(self):
        self.durations = []
        self.active_timing = False
        self.start_event = None
        self.end_event = None
    
    def start_timing(self):
        if self.active_timing:
            return  # Avoid nested timing with the same logger
        
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        self.active_timing = True
    
    def stop_timing(self):
        if not self.active_timing:
            return
        
        self.end_event.record()
        torch.cuda.synchronize()  # Wait for GPU operations to complete
        elapsed_time = self.start_event.elapsed_time(self.end_event)  # in milliseconds
        self.durations.append(elapsed_time)
        self.active_timing = False
    
    def time_block(self):
        # Return a context manager for convenient timing
        return TimingContextManager(self)
    
    def get_average_duration(self, clear_after=False, unit="milliseconds"):
        if not self.durations:
            return None
        
        avg_time = sum(self.durations) / len(self.durations)
        
        if unit == "seconds":
            avg_time /= 1000  # Convert from milliseconds to seconds
        
        if clear_after:
            self.durations = []
        
        return avg_time
    
    def get_last_duration(self, unit="milliseconds"):
        if not self.durations:
            return None
        
        last_time = self.durations[-1]
        
        if unit == "seconds":
            last_time /= 1000  # Convert from milliseconds to seconds
        
        return last_time

class TimingContextManager:
    def __init__(self, logger):
        self.logger = logger
    
    def __enter__(self):
        self.logger.start_timing()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.stop_timing()