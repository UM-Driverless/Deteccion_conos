#!/bin/bash

# SETUP PINMUX REGISTER to use vcan0 bus
# https://docs.nvidia.com/jetson/archives/r35.3.1/DeveloperGuide/text/HR/ControllerAreaNetworkCan.html
# sudo apt install busybox
# sends 0 as password to sudo, -S means superuser
echo 0 | sudo -S busybox devmem 0x0c303018 w 0x458 # vcan0 in
echo 0 | sudo -S busybox devmem 0x0c303010 w 0x400 # vcan0 out
echo 0 | sudo -S busybox devmem 0x0c303008 w 0x458 # can1 in
echo 0 | sudo -S busybox devmem 0x0c303000 w 0x400 # can1 out

# Setup the kernel drivers so vcan0 canbe recognized
echo 0 | sudo -S modprobe can
echo 0 | sudo -S modprobe can-raw
echo 0 | sudo -S modprobe mttcan

# CAN 0 RESET
sudo ip link set down vcan0 # Turn off in case it's on. Will return error the first time. Equivalent: sudo ifconfig vcan0 down
# sudo ip link set can0 type can bitrate 1000000 # 1Mbps
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0
# SEND MESSAGE 0x01234567 with ID = 0xABC
cansend vcan0 601#2F60600001
# LISTEN FOR MESSAGES
# echo "Message 0x01234567 with ID 123 sent. Now listening for messages..."
# sudo candump vcan0