#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@ Motorsport DV 2021
Script de escucha del bus CAN
"""

import can
import constants

class CANListener(object):

	def __init__(self, bus, log, datetime):

		self.bus = bus
		self.log = log
		self.datetime = datetime

		## Implementamos la clase abstracta Listener (https://python-can.readthedocs.io/en/master/listeners.html)
		# De momento usamos BufferedReader, que es una implementación por defecto de la librería, con un buffer de mensajes. Si necesitamos que sea async podemos
		# usar AsyncBufferedReader.
		self.buffer = can.BufferedReader()
		self.notifier = can.Notifier(self.bus, [self.buffer])
		with open(log, 'a', encoding="utf-8") as f:
		    f.write("[DEBUG]: Conectamos al listener" + '\n')


	def decode_message(self, msg):
		sender_id = msg.arbitration_id
		switcher = {
			# Rueda delantera izquierda
			constants.IMU_SENFL: self.get_sen_imu(0, msg.data),
			constants.SIG_SENFL: self.get_sen_signals(0, msg.data),
			constants.STATE_SENFL: "SEN STATE",
			# Rueda delantera derecha
			constants.IMU_SENFR: self.get_sen_imu(1, msg.data),
			constants.SIG_SENFR: self.get_sen_signals(1, msg.data),
			constants.STATE_SENFR: 0x305,
			# Rueda trasera izquierda
			constants.IMU_SENRL: self.get_sen_imu(2, msg.data),
			constants.SIG_SENRL: self.get_sen_signals(2, msg.data),
			constants.STATE_SENRL: 0x308,
			# Rueda trasera derecha
			constants.IMU_SENRR: self.get_sen_imu(3, msg.data),
			constants.SIG_SENRR: self.get_sen_signals(3, msg.data),
			constants.STATE_SENRR: 0x311,
			# ASSIS
			constants.ASSIS_C: 0x350,
			constants.ASSIS_R: 0x351,
			constants.ASSIS_L: 0x352,
			# ASB
			constants.ASB_ANALOG: 0x360,
			constants.ASB_SIGNALS: 0x361,
			constants.ASB_STATE: 0x362
		}
		return switcher.get(sender_id, lambda: "Código inválido")
		
	def get_sen_imu(self, wheel, data):
		acceleration_X = data[0]
		acceleration_Y = data[1]
		acceleration_Z = data[2]
		giroscope_X = data[3]
		giroscope_Y = data[4]
		giroscope_Z = data[5]
		magnet_X = data[6]
		magnet_Y = data[7]

		return [wheel, acceleration_X, acceleration_Y, acceleration_Z, giroscope_X, giroscope_Y, giroscope_Z, magnet_X, magnet_Y]

	def get_sen_signals(self, wheel, data):
		analog_1 = data[0]
		analog_2 = data[1]
		analog_3 = data[2]
		digital = data[3]

		speed_int = data[4]
		speed_decimals = data[5]
		speed = speed_int + speed_decimals/10**len(str(speed_decimals))

		revolutions_1 = data[6]
		revolutions_2 = data[7]

		revolutions = revolutions_2*16**2 + revolutions_1

		return [wheel, analog_1, analog_2, analog_3, digital, speed, revolutions]
