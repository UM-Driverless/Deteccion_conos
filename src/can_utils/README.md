# CAN communication utlities

Utilidades para la comunicación por protocolo CAN con el resto del vehículo.

Lo primero que hay que hacer es activar el CAN en el dispositivo (Jetson Xavier NX) ejecutando el archivo enable_CAN.sh
```bash
<path_to_Deteccion_conos>/can_scripts$ sh ./enable_CAN.sh
```

Para testear el correcto funcionamiento de los actuadores se encuentra el archivo car_actuator_testing.py en la carpeta principal del proyecto.
Es necesario tomar las medidas de seguridad para que el coche no se mueva, ya que este código solo se encarga de enviar las señales y no tiene ningún control sobre que está activado o desactivado en el coche.

```bash
<path_to_Deteccion_conos>$ python3 car_actuator_testing.py
```

Para configural el dispositivo (Jetson Xavier NX) para que al encenderlo arranque este código se proporciona el script run_actuator_testing.sh.
Habra que añadir al Startup Applications su ejecución a tarves del comando:

```bash
sh ./<path_to_Deteccion_conos>/run_can_on_startup.sh
```

Finalmente, el archivo enable_CAN_no_sudo solo funcionará en la Jetson Xavier.

# Activar y Desactivar
- Desactivar
    - `sudo ip link set can0 down`
    - `systemctl stop can-interface`