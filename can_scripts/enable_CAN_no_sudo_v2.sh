#!/bin/bash
echo driverless | sudo -S busybox devmem 0x0c303000 32 0x0000C400
echo driverless | sudo -S busybox devmem 0x0c303008 32 0x0000C458

echo driverless | sudo -S modprobe can
echo driverless | sudo -S modprobe can-raw
echo driverless | sudo -S modprobe mttcan
echo driverless | sudo -S ifconfig can0 down
echo driverless | sudo -S ip link set can0 up type can bitrate 1000000 berr-reporting on
#fd on
#cangen -v can0 -g 400

exit 0

