import time
import numpy as np

t0 = time.time()

a = []

for i in range(1000):
    a.append(1)

t1 = time.time()


b = np.array([])
for i in range(1000):
    b = np.append(b,1)
    
t2 = time.time()

print(f'TIMES: {t1-t0}  {t2-t1}')