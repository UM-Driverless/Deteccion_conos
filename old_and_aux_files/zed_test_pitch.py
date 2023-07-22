import pyzed
import pyzed.sl as sl
from time import sleep
import os
import math

zed = sl.Camera()

zed_params = sl.InitParameters()
zed_params.coordinate_units = sl.UNIT.METER
zed_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
zed_params.depth_mode = sl.DEPTH_MODE.NONE
zed.open(zed_params)
sensors = sl.SensorsData()

while True:
    sleep(0.001)
    zed.grab()
    os.system('clear')
    zed.get_sensors_data(sensors,sl.TIME_REFERENCE.CURRENT)
    print(f'Orientation: ',end=' ')
    quaternions = sensors.get_imu_data().get_pose().get_orientation().get()
    print(quaternions)
    print(f'PITCH: {math.atan2(2*quaternions[1]*quaternions[3] - 2*quaternions[0]*quaternions[2], 1 - 2*quaternions[1]**2 - 2 * quaternions[2]**2)}')
