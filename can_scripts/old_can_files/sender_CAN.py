#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@ Motorsport DV 2021
Script de arranque CAN
"""	

import can_scripts
from can_scripts import Message
from time import sleep
import constants
import math

class CANSender(object):

	def get_can_bus(self):
		return self.bus

	def __init__(self, bus, log, datetime):
		self.bus = bus
		self.log = log
		self.datetime = datetime

	def send_message(self, sender_id):
		msg = can_scripts.Message(arbitration_id = constants.TRAJECTORY_ACT, data=[0, 0, 0, 0, 1, 1, 1, 1], extended_id = False)
		with open(self.log, 'a', encoding="utf-8") as f:
			f.write("MSG: " + msg.__str__() + '\n')

		try:
			self.bus.send(msg)
		except can_scripts.CanError:
			with open(self.log, 'a', encoding="utf-8") as f:
				f.write("[ERROR]: Error al enviar" + '\n')
	def send_message(self, sender_id, throttle, brake, clutch, steer):
		sleep(0.5)
		msg = can_scripts.Message(arbitration_id = constants.TRAJECTORY_ACT, data=[math.trunc(throttle), math.trunc(brake), math.trunc(clutch), math.trunc(steer), 1, 0, 0, 0], extended_id = False)
		with open(self.log, 'a', encoding="utf-8") as f:
			f.write("MSG: " + msg.__str__() + '\n')

		try:
			self.bus.send(msg)
		except can_scripts.CanError:
			with open(self.log, 'a', encoding="utf-8") as f:
				f.write("[ERROR]: Error al enviar" + '\n')
