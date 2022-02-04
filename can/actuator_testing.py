from connection_utils.car_comunication import ConnectionManager
from controller_agent.testing_agent import AgentActuatorsTest as Agent
#from cone_detection.cone_segmentation import ConeDetector
from visualization_utils.visualizer_test_actuators import VisualizeActuators as Visualizer
import os
import time

# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# # os.environ["CUDA_DEVICE_ORDER"] = '0'

class VisualizerTest:
	def estimate_trajectory(cansender, image, in_speed, in_throttle, in_steer, in_brake, in_clutch, gear):
		verbose = 1

		# Inicializar detector de conos
		# detector = ConeDetector()

		# Inicializar conexiones
		connect_mng = ConnectionManager()

		# Inicializar Agente (controlador)
		agent = Agent()

		# Visualización de datos
		visualizer = Visualizer(max_data_to_store=10000)

		try:
			while True:
				start_time = time.time()

				# Detectar conos
				# detections, y_hat = detector.detect_cones(image)

				# Seleccionar acciones
				throttle, brake, steer, clutch, upgear, downgear = agent.get_action(1)
				print("Enviamos the actions")
				print("Datos: " + str(steer))

				# Enviar acciones
				connect_mng.send_actions(cansender, throttle=throttle,
						     brake=brake,
						     steer=steer,
						     clutch=clutch,
						     upgear=upgear,
						     downgear=downgear)

				print("FPS: ", 1.0 / (time.time() - start_time))

				#if verbose == 1:
					#visualizer.visualize([in_speed, in_throttle, in_steer, in_brake, in_clutch, gear], [throttle, brake, steer, clutch, upgear, downgear], print_can_data=True, print_agent_actions=True, real_time=True)

		finally:
			# Do whatever needed where the program ends or fails
			# connect_mng.close_connection()
			visualizer.visualize([in_speed, in_throttle, in_steer, in_brake, in_clutch, gear], [throttle, brake, steer, clutch, upgear, downgear], print_can_data=True, print_agent_actions=True, real_time=False)

		visualizer.close_windows()
