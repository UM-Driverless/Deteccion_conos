sudo apt-get remove --purge nvidia-* -y
sudo apt autoremove
sudo ubuntu-drivers autoinstall
sudo service lightdm restart
sudo apt install nvidia-driver-515 nvidia-dkms-515 # This is the driver installed
spd-say -t male1 -l es "He acabado, chachos. Ahora me reinicio"
sleep 4
sudo reboot