import matplotlib.pyplot as plt
import numpy as np

fontsize = 30
fig = plt.figure(1, figsize=(20, 4))
x = [i for i in range(400)]

epsilon_init = 1.0
epsilon = 1.
y = []
for i in range(400):
    y.append(epsilon)
    epsilon = epsilon*0.99


ax = fig.add_subplot(1, 3, 1)
ax.plot(x, y)
ax.set_title('Single annealing', fontsize=fontsize)
ax.set_ylabel('Epsilon', fontsize=fontsize)
plt.xticks([])
plt.yticks(fontsize=int(fontsize*0.6))

epsilon_init = 1.0
epsilon = 1.
y = []
for i in range(400):
    y.append(epsilon)
    epsilon = epsilon*0.95
    if i % 100==0:
        epsilon = epsilon_init

ax = fig.add_subplot(1, 3, 2)
ax.plot(x, y)
ax.set_title('Double annealing', fontsize=fontsize)
ax.set_xlabel('Iterations', fontsize=fontsize)
plt.xticks([])

epsilon_init = 1.0
epsilon = 1.
y = []
for i in range(400):
    y.append(epsilon)
    epsilon = epsilon*0.95
    if i % 100==0:
        epsilon_init = epsilon_init - 0.2
        epsilon = epsilon_init

ax = fig.add_subplot(1, 3, 3)
ax.plot(x, y)
ax.set_title('Multiple annealing', fontsize=fontsize)
plt.xticks([])
plt.show()
