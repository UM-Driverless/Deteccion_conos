# How to install and use the scripts in linux (Tested with Ubuntu 22)

- Create a conda environment:
    ```bash
    conda create -n FormulaStudent -y
    ```
    (en caso de no tener conda, instalarlo siguiendo los pasos de esta web: 
    https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
- Activate the environment
    ```bash
    conda activate FormulaStudent
    ```
- Install git, and pip
    ```bash
    sudo apt install git
    conda install pip
    
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
    ```
- [OPTIONAL] If you want to modify the weights, include the [weights folder](https://urjc-my.sharepoint.com/:f:/r/personal/r_jimenezm_2017_alumnos_urjc_es/Documents/formula/formula%2022-23/SOFTWARE/FILES/yolov5_models?csf=1&web=1&e=nILHR5) in: `"yolov5/weights"`
- ZED Camera Installation
    1. Download the SDK according to desired CUDA version and system (Ubuntu, Nvidia jetson xavier jetpack, ...)
        https://www.stereolabs.com/developers/release/
    2. Activate the conda environment (To be safe. Keep the ZED's Python API inside the environment)
        ```bash
        conda activate FormulaStudent
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
    7. To get the Python API:
        ```bash
        python3 /usr/local/zed/get_python_api.py 
        ```
- You should be able to run:
    ```bash
    python3 own_camera_main_no_can.py
    ```
* To explore if something fails:
    * `sudo apt-get install python3-tk`

# Cliente para realizar la detección de conos en el simulador

Este cliente funciona en conjunto con el simulador desarrollado en https://github.com/AlbaranezJavier/UnityTrainerPy. Para hacerlo funcionar solo será necesario seguir las instrucciones del repositorio indicado para arrancar el simulador y posteriormente ejecutar el cliente que podemos encontrar en el archivo /PyUMotorsport/main_cone_detection.py

Los pesos de la red neuronal para el main.py se encuentran en el siguiente enlace: https://drive.google.com/file/d/1H-KOYKMu6KM3g8ENCnYPSPTvb6zVnnFX/view?usp=sharing
Se debe descomprimir el archivo dentro de la carpeta: /PyUMotorsport/cone_detection/saved_models/

Los pesos de la red neuronal para el main_2.py se encuentran en el siguiente enlace: https://drive.google.com/file/d/1NFDBKxpRcfPs8PV3oftLya_M9GxW8O5h/view?usp=sharing
Se debe descomprimir el archivo dentro de la carpeta: /PyUMotorsport_v2/ObjectDetectionSegmentation/DetectionData/

# Instalación

Crea tu entorno virtual en python 3.8 y activalo
```bash
conda create -n FormulaStudent python=3.8
conda activate FormulaStudent
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

