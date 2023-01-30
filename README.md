# How to install and use the scripts in linux (Tested with Ubuntu 22)
- If conda is not installed: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
- Create a conda environment:
    ```bash
    conda create -n formula -y
    
    # To remove it:
    # conda env remove -n formula
    ```
- Activate the environment
    ```bash
    conda activate formula
    ```
- Install git, pip, numpy, update the compiler
    ```bash
    sudo apt install git
    conda install pip
    conda install numpy # With conda instead of pip, otherwise odd errors
    conda install -c conda-forge gcc=12.1.0 # Otherwise zed library throws error: version `GLIBCXX_3.4.30' not found
    
    # If outside of conda environment:
    # sudo apt install python3-pip
    ```
- Clone the GitHub directory:
    ```bash
    git clone https://github.com/UM-Driverless/Deteccion_conos.git
    git checkout Test_Portatil
    ```
- Install the requirements (for yolo network and for our scripts)
    ```bash
    cd ~/Deteccion_conos
    pip install -r requirements.txt
    
    # To do it with conda:
    # conda install --file requirements.txt
    ```
    The requirements file should be installed with conda to prevent compatibility problems, but many packages are not available in conda, so are installed with pip
- [OPTIONAL] If you want to modify the weights, include the [weights folder](https://urjc-my.sharepoint.com/:f:/r/personal/r_jimenezm_2017_alumnos_urjc_es/Documents/formula/formula%2022-23/SOFTWARE/FILES/yolov5_models?csf=1&web=1&e=nILHR5) in: `"yolov5/weights"`
- ZED Camera Installation
    1. Download the SDK according to desired CUDA version and system (Ubuntu, Nvidia jetson xavier jetpack, ...)
        https://www.stereolabs.com/developers/release/
    2. Activate the conda environment (To be safe. Keep the ZED's Python API inside the environment)
        ```bash
        conda activate formula
        ```
    3. Add permits:
        ```bash
        sudo chmod 777 {FILENAME}
        ```
    4. Run it without sudo (You can copy the file and Ctrl+Shift+V into the terminal. Don't know why tab doesn't complete the filename):
        ```bash
        sh {FILENAME}.run
        ```
    5. By default accept to install cuda, static version of SDK, AI module, samples and **Python API**. Diagnostic not required.
    6. Now it should be installed in the deault installation path: `/usr/local/zed`
    7. To get the Python API (Otherwise pyzed won't be installed and will throw an error):
        ```bash
        python3 /usr/local/zed/get_python_api.py 
        ```
- You should be able to run:
    ```bash
    python3 own_camera_main_no_can.py
    ```
* To explore if something fails:
    * `sudo apt-get install python3-tk`

# NVIDIA JETSON XAVIER NX SETUP
TODO Testing with Jetpack 5.1
- [Prepare Xavier](https://developer.nvidia.com/embedded/learn/get-started-jetson-xavier-nx-devkit#prepare)
    - Takes about 30 min
- Install ZED camera drivers
    - [ZED SDK for L4T 35.1 (Jetpack 5.0)](https://download.stereolabs.com/zedsdk/3.8/l4t35.1/jetsons)
    - (https://www.stereolabs.com/developers/release/)
    - [Python API](https://www.stereolabs.com/docs/app-development/python/install/)
    - (Test Record: https://github.com/SusanaPineda/utils_zed/blob/master/capture_loop.py)
- Startup script (Setup all the programs on startup)
    - Add in Startup Applications: "python3 startup_script.py"
- (CAN: https://medium.com/@ramin.nabati/enabling-can-on-nvidia-jetson-xavier-developer-kit-aaaa3c4d99c9)

# Cliente para realizar la detección de conos en el simulador

Este cliente funciona en conjunto con el simulador desarrollado en https://github.com/AlbaranezJavier/UnityTrainerPy. Para hacerlo funcionar solo será necesario seguir las instrucciones del repositorio indicado para arrancar el simulador y posteriormente ejecutar el cliente que podemos encontrar en el archivo /PyUMotorsport/main_cone_detection.py

Los pesos de la red neuronal para el main.py se encuentran en el siguiente enlace: https://drive.google.com/file/d/1H-KOYKMu6KM3g8ENCnYPSPTvb6zVnnFX/view?usp=sharing
Se debe descomprimir el archivo dentro de la carpeta: /PyUMotorsport/cone_detection/saved_models/

Los pesos de la red neuronal para el main_2.py se encuentran en el siguiente enlace: https://drive.google.com/file/d/1NFDBKxpRcfPs8PV3oftLya_M9GxW8O5h/view?usp=sharing
Se debe descomprimir el archivo dentro de la carpeta: /PyUMotorsport_v2/ObjectDetectionSegmentation/DetectionData/

# Old (install tensorflow detection)

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

## Actualizar Xavier para ejecutar YOLOv5 (06/2022)

```bash
git clone https://github.com/UM-Driverless/Deteccion_conos.git
cd Deteccion_conos
pip3 install -r yolov5/yolo_requeriments.txt
sh can_scripts/enable_CAN.sh
python3 car_actuator_testing_zed_conect_yolo.py 
```

