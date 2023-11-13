import time
class Chronometer:
    def __init__(self):
        self.elapsed_time = 0
    
    def begin_time_window(self):
        self.start_time = time.perf_counter()
    
    def end_time_window(self):
        self.elapsed_time += time.perf_counter() - self.start_time

    def get_total_time(self):
        print("Elapsed total time:", self.elapsed_time)