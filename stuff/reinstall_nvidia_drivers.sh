sudo apt-get remove --purge nvidia-* -y
sudo apt autoremove
sudo ubuntu-drivers autoinstall -y
sudo service lightdm restart
sudo apt install nvidia-driver-515 nvidia-dkms-515
spd-say -t male1 -l es "He acabado, chachos. Ahora me reinicio"
sleep 1
sudo reboot