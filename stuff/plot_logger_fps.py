import matplotlib.pyplot as plt
import numpy as np
import re

# Read data from the file
file_path = "2023_07_30__15_54.txt"
iterations = []
fps_values = []

with open(file_path, 'r') as file:
    for i, line in enumerate(file):
        try:            
            iteration = int(re.search('Iter (\d+)',line).group(1))
            fps_value = float(re.search('FPS: ([\d\.-]+)',line).group(1))
            
            iterations.append(iteration)
            fps_values.append(fps_value)
            
        except Exception as e:
            print(f"Ignoring line {i}. (Error {e}")

# Plot the data as a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(iterations, fps_values, s=4)
# plt.plot(iterations, fps_values, linestyle='-')

# plt.xlabel("Iterations")
# plt.ylabel("FPS (Frames Per Second)")
# plt.title("FPS vs. Iteration (Scatter Plot)")
plt.grid(True)
plt.tight_layout()
plt.show()

print(f'AVERAGE FPS: {np.average(fps_values)}')