#!/bin/bash
echo xavier | sudo -S busybox devmem 0x0c303000 32 0x0000C400
echo xavier | sudo -S busybox devmem 0x0c303008 32 0x0000C458

echo xavier | sudo -S modprobe can
echo xavier | sudo -S modprobe can-raw
echo xavier | sudo -S modprobe mttcan
echo xavier | sudo -S ifconfig can0 down
echo xavier | sudo -S ip link set can0 up type can bitrate 1000000 berr-reporting on
#fd on
#cangen -v can0 -g 400

exit 0

