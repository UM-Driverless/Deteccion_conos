import matplotlib.pyplot as plt

# Read data from the file
file_path = "logs/2023_07_29__20_36.txt"
iterations = []
fps_values = []

with open(file_path, 'r') as file:
    for line in file:
        # print(line)
        parts = line.strip().split(": ")
        # print(parts)
        try:
            iteration = int(float(parts[1].split(" ")[1]))
            fps_value = float(parts[2])
            iterations.append(iteration)
            fps_values.append(fps_value)
        except Exception as e:
            print(f"Error processing line: {line.strip()}\n{e}\n\n")

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(iterations, fps_values, linestyle='-', color='b')
plt.xlabel("Iteration")
plt.ylabel("FPS (Frames Per Second)")
plt.title("FPS vs. Iteration")
plt.grid(True)
plt.tight_layout()
plt.show()
