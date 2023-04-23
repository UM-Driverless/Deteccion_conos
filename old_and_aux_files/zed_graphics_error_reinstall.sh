sudo apt-get remove --purge nvidia-* -y
sudo apt autoremove
sudo ubuntu-drivers autoinstall
sudo service lightdm restart
sudo apt install nvidia-driver-515 nvidia-dkms-515
sudo reboot
sudo spd-say "He acabado chachos"