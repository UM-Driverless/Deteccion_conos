#!/bin/bash
sudo busybox devmem 0x0c303000 32 0x0000C400
sudo busybox devmem 0x0c303008 32 0x0000C458

sudo modprobe can
sudo modprobe can-raw
sudo modprobe mttcan
sudo ifconfig vcan0 down
#sudo ip link set can0 up type can bitrate 1000000 restart-ms 100 berr-reporting on
sudo ip link set vcan0 up type can bitrate 1000000
#sudo ifconfig can0 txqueuelen 1000
#cangen -v can0 -g 400

exit 0

