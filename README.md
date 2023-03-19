# How to install and use the scripts in linux (Tested with Ubuntu 22)
We won't use Conda since it's not necessary, and the several python versions have caused problems. Also conda can't install all the packages we need, so there would be some packages installed with pip and others with conda. It also caused problems with docker.

- First apt installs
    ```bash
    sudo apt update && sudo apt upgrade -y; spd-say "I finished the update"
    sudo apt install curl nano git pip python3 zstd #zstd is zed dependency; spd-say "I finished installing"
    ```
- Clone the GitHub directory:
    ```bash
    git clone https://github.com/UM-Driverless/Deteccion_conos.git
    ```
- Install the requirements (for yolo network and for our scripts)
    ```bash
    cd ~/Deteccion_conos
    pip install -r <requirements_file_name>.txt
    
    # To do it with conda:
    # conda install --file requirements.txt
    ```
- [OPTIONAL] If you want to modify the weights, include the [weights folder](https://urjc-my.sharepoint.com/:f:/r/personal/r_jimenezm_2017_alumnos_urjc_es/Documents/formula/formula%2022-23/SOFTWARE/FILES/yolov5_models?csf=1&web=1&e=nILHR5) in: `"yolov5/weights"`

- ZED Camera Installation
    1. Download the SDK according to desired CUDA version and system (Ubuntu, Nvidia jetson xavier jetpack, ...)
        https://www.stereolabs.com/developers/release/
    2. Add permits:
        ```bash
        sudo chmod 777 {FILENAME}
        ```
    3. Run it without sudo (You can copy the file and Ctrl+Shift+V into the terminal. Don't know why tab doesn't complete the filename):
        ```bash
        sh {FILENAME}.run
        ```
    4. By default accept to install cuda, static version of SDK, AI module, samples and **Python API**. Diagnostic not required.
    5. Now it should be installed in the deault installation path: `/usr/local/zed`
    6. To get the Python API (Otherwise pyzed won't be installed and will throw an error):
        ```bash
        python3 /usr/local/zed/get_python_api.py
        ```
- You should be able to run:
    ```bash
    python3 main.py
    ```
- To make sure you are using the GPU (Get IS CUDA AVAILABLE? : True)
    - Check the GPU driver. X.Org -> nvidia-driver-515. In Software and Updates.
    - If errors:
        ```bash
        sudo apt-get remove --purge nvidia-* -y
        sudo apt autoremove
        sudo ubuntu-drivers autoinstall
        sudo service lightdm restart
        sudo apt install nvidia-driver-515 nvidia-dkms-515
        sudo reboot
        ```
* To explore if something fails:
    * `sudo apt-get install python3-tk`

#INSTALL SIMULATOR
    Go to https://github.com/FS-Driverless/Formula-Student-Driverless-Simulator/releases and download the latest version. The file has an UNREAL ENGINE       file which executes the simulator.
    
# NVIDIA JETSON XAVIER NX SETUP
TODO Testing with Jetpack 5.1
- Start here to install the OS: [https://developer.nvidia.com/embedded/learn/get-started-jetson-xavier-nx-devkit](https://developer.nvidia.com/embedded/learn/get-started-jetson-xavier-nx-devkit)
    - Takes about 1h.
    - Prepare SD card with >=32GB, a way to connect it to a computer (sd to usb adapter), fast internet, 
    - First download the [Jetson Xavier NX Developer Kit SD Card Image](https://developer.nvidia.com/embedded/jetpack). Older versions [here](https://developer.nvidia.com/embedded/jetpack-archive).
        - JetPack 5.1 is the latest version. JetPack 5.0.2 is the latest with docker pytorch installation available, and it's the one we've used.
        - JetPack 4.5.1 works with Pytorch 1.8 according to https://cognitivexr.at/blog/2021/03/11/installing-pytorch-and-yolov5-on-an-nvidia-jetson-xavier-nx.html
    - Then you'll be asked to install "SD Card formatter" and "Etcher"
    - Follow the tutorial for the rest
- Set power mode (up right in task bar) to max
- First apt installs
    ```bash
    sudo apt update && sudo apt upgrade -y; spd-say "I finished installing"
    sudo apt install curl nano git zstd #zstd is zed dependency
    ```
- Clone the GitHub directory:
    ```bash
    git clone https://github.com/UM-Driverless/Deteccion_conos.git; spd-say "Done cloning the repository"
    ```
- To make bluetooth work [link](https://forums.developer.nvidia.com/t/bt-cant-work-after-flash-jetson-tx2-with-jetpacke4-3-no-bt-speaker-and-keyboard/112098/4):
    1. Navigate to the following file:
    $ sudo vim /lib/systemd/system/bluetooth.service.d/nv-bluetooth-service.conf
    2. Search for below line:
    ExecStart=/usr/lib/bluetooth/bluetoothd -d --noplugin=audio,a2dp,avrcp
    3. Remove all options for no plugin. It should look like below:
    ExecStart=/usr/# KVASER Setup in Ubuntu
- Follow the tutorial: https://cognitivexr.at/blog/2021/03/11/installing-pytorch-and-yolov5-on-an-nvidia-jetson-xavier-nx.html
    - Script that automatically installs everything
        ```bash
        curl https://raw.githubusercontent.com/cognitivexr/edge-node/main/scripts/setup-xavier.sh | bash
        ```
    - To solve `Illegal instruction (core dumped)`, issue with numpy and openblas:
        ```bash
        pip3 install -U "numpy==1.19.4"
        ```


- Install ZED camera drivers
    - [ZED SDK for L4T 35.1 (Jetpack 5.0)](https://download.stereolabs.com/zedsdk/3.8/l4t35.1/jetsons)
    - (https://www.stereolabs.com/developers/release/)
    - [Python API](https://www.stereolabs.com/docs/app-development/python/install/)
    - (Test Record: https://github.com/SusanaPineda/utils_zed/blob/master/capture_loop.py)
    - If shared library error:
        ```bash
        sudo apt install libturbojpeg0-dev
        ```
- Startup script (Setup all the programs on startup)
    - Add in Startup Applications: "python3 startup_script.py"
- (CAN: https://medium.com/@ramin.nabati/enabling-can-on-nvidia-jetson-xavier-developer-kit-aaaa3c4d99c9)

```bash
tar -xvzf linuxcan.tar.gz
sudo apt-get install build-essential
sudo apt-get install linux-headers-`uname -r`
```
In linuxcan, and linuxcan/canlib, run:
```bash
make
sudo make install
```
In linuxcan/common, run:
```bash
make
sudo ./installscript.sh
```
To DEBUG:
```bash
make KV_Debug_ON=1
```
    1. Then update with below command:
    $ sudo apt-get update
    2. Install pulse audio using below command:
    $ sudo apt-get install pulseaudio-module-bluetooth
    3. Reboot device as per below command:
    $ sudo reboot
    4. Pair and use your Bluetooth audio devices.

- To use:
    - First plug power, then the HDMI port, because otherwise it doesn't turn on
    - Don't use the upper left USB-A port for high speed (ZED camera). It's 2.0 while the others are 3.1

# KVASER Setup in Ubuntu
- Reference: https://www.kvaser.com/linux-drivers-and-sdk/
- Video: https://www.youtube.com/watch?v=Gz-lIVIU7ys
- SDK: https://www.kvaser.com/downloads-kvaser/?utm_source=software&utm_ean=7330130980754&utm_status=latest

```bash
tar -xvzf linuxcan.tar.gz
sudo apt-get install build-essential
sudo apt-get install linux-headers-`uname -r`
```
In linuxcan, and linuxcan/canlib, run:
```bash
make
sudo make install
```
In linuxcan/common, run:
```bash
make
sudo ./installscript.sh
```
To have the python API:
```bash
pip3 install canlib
```
To DEBUG:
```bash
make KV_Debug_ON=1
```


# Cliente para realizar la detección de conos en el simulador

Este cliente funciona en conjunto con el simulador desarrollado en https://github.com/AlbaranezJavier/UnityTrainerPy. Para hacerlo funcionar solo será necesario seguir las instrucciones del repositorio indicado para arrancar el simulador y posteriormente ejecutar el cliente que podemos encontrar en el archivo /PyUMotorsport/main_cone_detection.py

Los pesos de la red neuronal para el main.py se encuentran en el siguiente enlace: https://drive.google.com/file/d/1H-KOYKMu6KM3g8ENCnYPSPTvb6zVnnFX/view?usp=sharing
Se debe descomprimir el archivo dentro de la carpeta: /PyUMotorsport/cone_detection/saved_models/

Los pesos de la red neuronal para el main_2.py se encuentran en el siguiente enlace: https://drive.google.com/file/d/1NFDBKxpRcfPs8PV3oftLya_M9GxW8O5h/view?usp=sharing
Se debe descomprimir el archivo dentro de la carpeta: /PyUMotorsport_v2/ObjectDetectionSegmentation/DetectionData/

## To test
Go to canlib/examples
```bash
./listChannels
./canmonitor 0
```

## To install any driver (canlib and kvcommon must be installed first):
```bash
make
sudo ./installscript.sh
```

---
# Old stuff

Crea tu entorno virtual en python 3.8 y activalo
```bash
conda create -n formula python=3.8
conda activate formula
#conda install tensorflow-gpu
```

[comment]: <> (&#40;pip install -r requeriments.txt&#41;)

A continuación vamos a installar el Model Zoo de detección de Tensorflow

Si no tienes todavía la carpeta models/research/
```bash
git clone --depth 1 https://github.com/tensorflow/models
```

Una vez dispones de la carpeta models/research/

```bash
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```

Actualizar Xavier para ejecutar YOLOv5 (06/2022)
```bash
git clone https://github.com/UM-Driverless/Deteccion_conos.git
cd Deteccion_conos
pip3 install -r yolov5/yolo_requeriments.txt
sh can_scripts/enable_CAN.sh
python3 car_actuator_testing_zed_conect_yolo.py
```

- Try to use a preconfigured JetPack 5.0.2 PyTorch Docker container, with all the dependencies and versiones solved: https://blog.roboflow.com/deploy-yolov5-to-jetson-nx/
    - Register in docker website
    - Login. If it doesn't work, reboot and try again.
        ```bash
        docker login
        ```
    - Take the tag of a container from here: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch . For example, for JetPack 5.0.2 (L4T R35.1.0) it's `l4t-pytorch:r35.1.0-pth1.13-py3`
    - Pull container
        ```bash
        # l4t-pytorch:r35.1.0-pth1.13-py3 ->
        sudo docker pull nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.13-py3
        ```
    - Run container
        ```bash
        # Will download about 10GB of stuff
        sudo docker run -it --rm --runtime nvidia --network host nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.13-py3
        ```
    - TODO FINISH


(Install visual studio, pycharm, telegram, ...)
