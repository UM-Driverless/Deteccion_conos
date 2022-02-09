#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@ Motorsport DV 2021
Script de arranque CAN
"""	

import sys
sys.path.append('/home/xavier/CAN/Deteccion_conos-main/')

import datetime
import constants
import can_scripts
from listener_CAN import CANListener
from sender_CAN import CANSender
from can_scripts.actuator_testing import VisualizerTest

def main():
	## Fichero de logs
	path = "/home/xavier/CAN/logs/"
	date = datetime.datetime.now()
	log = path + date.strftime("%Y_%m_%d__%H_%M_") + 'LOGS' + '.txt'

	## Inicializamos el bus CAN
	# Hay que ver qué interfaz usamos o si nos vale socketcan por defecto
	# El canal es el can_scripts físico (can0) como puede verse al ejecutar 'ifconfig' en la consola
	# El bitrate debe ser igual al definido en el script enable_CAN.sh
	bus = can_scripts.interface.Bus(bustype='socketcan', channel='can0', bitrate=1000000)
	with open(log, 'a', encoding="utf-8") as f:
	    f.write("[DEBUG]: Conectado al bus " + bus.channel_info + " [" + str(datetime.datetime.now()) + "]" + '\n')

	canlistener = CANListener(bus, log, datetime)
	cansender = CANSender(bus, log, datetime)
	#while True:
	#	cansender.send_message(302)

	## Polling de mensajes
	buenos = 0
	with open(log, 'a', encoding="utf-8") as f:
	    		f.write("[DEBUG]: llamamos al visu " + '\n')
	VisualizerTest.estimate_trajectory(cansender, [], 40.5, 0, 0, 0, 0, 0)
	##while True:

	msg = canlistener.buffer.get_message()
	if msg is not None:
		with open(log, 'a', encoding="utf-8") as f:
	    		f.write("[DEBUG]: Mensaje bueno recibido (" + str(buenos+1) + ") -> " + msg.__str__() + '\n')
		buenos = buenos + 1
		
		message = canlistener.decode_message(msg)
		with open(log, 'a', encoding="utf-8") as f:
			f.write("[DEBUG]: Mensaje leído (" + str(message) + ") " + '\n')
		## Llamada a la red neuronal
		if msg.arbitration_id == constants.SIG_SENFL or msg.arbitration_id == constants.SIG_SENFR or msg.arbitration_id == constants.SIG_SENRL or msg.arbitration_id == constants.SIG_SENRR:
			VisualizerTest.estimate_trajectory(cansender, [], message[5], 0, 0, 0, 0, 0)

		#cansender.send_message(msg.arbitration_id)

			

if __name__ == "__main__":
    main()
