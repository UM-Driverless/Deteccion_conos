# This code adds the fsds package to the pyhthon path.
# It assumes the fsds repo is cloned in the home directory.
# Replace fsds_lib_path with a path to wherever the python directory is located.
import sys, os
fsds_lib_path = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python")
sys.path.insert(0, fsds_lib_path)

import time
import fsds

# connect to the AirSim simulator 
client = fsds.FSDSClient()

# Check network connection
client.confirmConnection()

# After enabling api controll only the api can controll the car. 
# Direct keyboard and joystick into the simulator are disabled.
# If you want to still be able to drive with the keyboard while also 
# controll the car using the api, call client.enableApiControl(False)
client.enableApiControl(True)

# Instruct the car to go full-speed forward
car_controls = fsds.CarControls()
car_controls.throttle = 1
client.setCarControls(car_controls)

time.sleep(5)

# Places the vehicle back at it's original position
client.reset()
