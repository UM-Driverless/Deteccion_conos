import pyzed
import pyzed.sl as sl
from time import sleep
import os
import matplotlib.pyplot as plt
import numpy as np

zed = sl.Camera()
i = sl.InitParameters()
i.depth_mode = sl.DEPTH_MODE.NONE
zed.open(i)
sensors = sl.SensorsData()

orientation_data = []

def normalize_quaternion(q):
    norm = np.linalg.norm(q)
    return q / norm

def plot_orientation_data():
    plt.clf()
    orientation_data_np = np.array(orientation_data)
    for i in range(4):  # 4 elements in the quaternion (w, x, y, z)
        plt.plot(orientation_data_np[:, i], label=f'Orientation {i}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Orientation Data')
    plt.legend()
    plt.grid()
    plt.pause(0.001)

plt.ion()  # Turn on interactive mode for matplotlib

try:
    while True:
        sleep(0.001)
        zed.grab()
        os.system('clear')
        zed.get_sensors_data(sensors, sl.TIME_REFERENCE.CURRENT)

        orientation = sensors.get_imu_data().get_pose().get_orientation().get()
        normalized_orientation = normalize_quaternion(orientation)
        orientation_data.append(normalized_orientation)

        print(f'Orientation: ', end=' ')
        for i in orientation:
            print(f'{i:.5f}', end='\t')
        print()
        print(f'LinearAcceleration: ', end=' ')
        for i in sensors.get_imu_data().get_linear_acceleration():
            print(f'{i:.5f}', end='\t')
        print('[m/sec^2]')
        print(f'AngularVelocity: ', end=' ')
        for i in sensors.get_imu_data().get_angular_velocity():
            print(f'{i:.5f}', end='\t')
        print('[deg/sec]')

        plot_orientation_data()

except KeyboardInterrupt:
    zed.close()
    print("ZED camera closed.")
