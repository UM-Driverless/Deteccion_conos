from connection_utils.communication_controllers import can_utils
import time

can_utils.CAN.get_rpm_can()
time.sleep(1)

can_utils.CAN.get_speed_FL_can()
time.sleep(1)

can_utils.CAN.get_speed_FR_can()
time.sleep(1)

can_utils.CAN.get_amr()
time.sleep(1)

can_utils.CAN.get_ASState()
time.sleep(1)

can_utils.CAN.get_brake_pressure()
time.sleep(1)

can_utils.CAN.get_clutch_state()
time.sleep(1)

can_utils.CAN.get_steer_angle()
time.sleep(1)

can_utils.CAN.get_throttle_pos()
time.sleep(1)