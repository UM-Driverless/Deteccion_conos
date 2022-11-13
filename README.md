Instrucciones de instalación en linux:

1. Crea un entorno de conda:
conda create -n FormulaStudent -y
(en caso de no tener conda, instalarlo siguiendo los pasos de esta web: 
https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

```bash
sudo apt install git &
sudo apt install python3-pip
```


2. Activa el entorno
conda activate FormulaStudent

3. clona el directorio de github:
git clone https://github.com/UM-Driverless/Deteccion_conos.git
git checkout Test_Portatil

4. Instala los requisitos
cd ~/Deteccion_conos
pip3 install -r yolov5/yolo_requeriments.txt

5. Instala las dos librerías restantes:
pip install simple-pid
pip install python-can

6. Descomprime la carpeta de los pesos en: "yolov5/weights"
ruta de los pesos: https://urjc-my.sharepoint.com/:u:/r/personal/r_jimenezm_2017_alumnos_urjc_es/Documents/formula/formula%2022-23/SOFTWARE/FILES/yolov5_models.zip?csf=1&web=1&e=EWceWu

7. Debería serte posible ejecutar el siguiente comando:
python3 own_camera_main_no_can.py



--------------------------------------------------------------------------------------------------------------------------------------------------------------------

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
conda install tensorflow-gpu
conda install pandas
pip install opencv-python
pip install simple-pid
conda install pillow
conda install matplotlib
pip install opencv-contrib-python

```

[comment]: <> (&#40;pip install -r requeriments.txt&#41;)

A continuación vamos a installar el Model Zoo de detección de Tensorflow

Si no tienes todavía la carpeta models/research/
```bash
git clone --deph 1 https://github.com/tensorflow/models
```

Una vez dispones de la carpeta models/research/

```bash
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```

## Instalación ZED
Descarga la versión del SDK que corresponda según la versión de CUDA y el sistema operativo (o Jetpack) en la página de Stereolabs:
https://www.stereolabs.com/developers/release/

En la carpeta de descargas ejecuta:

```bash
sh ./ZED_SDK_Ubuntu20_cuda11.5_v3.6.5.run
```

Sigue los pasos del instalador aceptando la instalacion de la API de python.

## Actualizar Xavier para ejecutar YOLOv5 (06/2022)

```bash
git clone https://github.com/UM-Driverless/Deteccion_conos.git
cd Deteccion_conos
pip3 install -r yolov5/yolo_requeriments.txt
sh can_scripts/enable_CAN.sh
python3 car_actuator_testing_zed_conect_yolo.py 
```

