import time
from tools.time_counter import Time_Counter


timer = Time_Counter()
while True:
    timer.add_time()
    time.sleep(10e-3)
    timer.add_time()
    time.sleep(20e-3)
    timer.add_time()
    time.sleep(10e-3)
    timer.add_time()
    
    timer.new_iter()