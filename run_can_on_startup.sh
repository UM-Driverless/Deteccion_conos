#!/bin/bash
sh ./Deteccion_conos/can/enable_CAN_no_sudo.sh
python3 /Deteccion_conos/car_actuator_testing.py
exit 0

