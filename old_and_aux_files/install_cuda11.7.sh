wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-11-7
spd-say -t male1 -l es "CUDA 1.7 instalado. Ahora me reinicio"
sleep 4
sudo reboot