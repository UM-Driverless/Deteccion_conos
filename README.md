# Cliente para realizar la detección de conos en el simulador

Este cliente funciona en conjunto con el simulador desarrollado en https://github.com/AlbaranezJavier/UnityTrainerPy. Para hacerlo funcionar solo será necesario seguir las instrucciones del repositorio indicado para arrancar el simulador y posteriormente ejecutar el cliente que podemos encontrar en el archivo /PyUMotorsport/main_cone_detection.py

Los pesos de la red neuronal  se encuentran en el siguiente enlace: https://drive.google.com/file/d/1H-KOYKMu6KM3g8ENCnYPSPTvb6zVnnFX/view?usp=sharing
Se debe descomprimir el archivo dentro de la carpeta: /PyUMotorsport/cone_detection/saved_models/

# Instalación

Crea tu entorno virtual en python 3.8 y activalo
```bash
pip install -r requeriments
```

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
