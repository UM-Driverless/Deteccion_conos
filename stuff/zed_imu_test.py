import pyzed
import pyzed.sl as sl
from time import sleep
import os
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
    for zed_params in sensors.get_imu_data().get_pose().get_orientation().get():
        print(f'{zed_params:.5f}',end='\t')
    print()
    print(f'LinearAcceleration: ',end=' ')
    for zed_params in sensors.get_imu_data().get_linear_acceleration():
        print(f'{zed_params:.5f}',end='\t')
    print('[m/sec^2]')
    print(f'AngularVelocity: ',end=' ')
    for zed_params in sensors.get_imu_data().get_angular_velocity():
        print(f'{zed_params:.5f}',end='\t')
    print('[deg/sec]')