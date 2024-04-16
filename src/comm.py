import os, sys, cv2, time
import numpy as np
import multiprocessing
from abc import ABC, abstractmethod

class CarCommunicator(ABC):
    @abstractmethod
    def send_actuation(self, actuation):
        """Sends actuation commands (steering, throttle, etc.) to the car."""
        pass
    
    @abstractmethod
    def receive_state(self, state):
        """Receives sensor data (speed, RPM, etc.) from the car."""
        pass
    
    @staticmethod
    def create(mode):
        """Factory method to create a CarCommunicator based on the mode."""
        if mode == 'off' or mode == False:
            return NullCarCommunicator()
        elif mode == 'serial':
            return SerialCommunicator('/dev/ttyUSB0')
        elif mode == 'can_jetson':
            return CanJetson('can0')
        elif mode == 'can_kvaser':
            return CanKvaser('kvaser')
        else:
            raise ValueError(f'Unknown comm mode: {mode}')
        
class NullCarCommunicator(CarCommunicator):
    def send_actuation(self, actuation):
        # Do nothing (no actuation sent)
        pass

    def receive_state(self, state):
        # Return empty state data (no data received)
        return {}


class SerialCommunicator(CarCommunicator):
    """
    TODO IMPLEMENT THIS FOR ROBOT
    """
    def __init__(self, port):
        self.serial_port = serial.Serial(port)  # Initialize serial connection

    def send_actuation(self, actuation):
        # Convert actuation data to a format suitable for serial communication
        # ... (implementation specific to serial protocol)
        self.serial_port.write(data_to_send)

    def receive_state(self, state):
        # Read data from serial port and parse it into sensor data
        # ... (implementation specific to serial protocol)
        return sensor_data

class CanJetson(CarCommunicator):
    def __init__(self, can_bus):
        from can_utils.can_utils import CAN
        self.can_bus = can_bus  # Initialize CAN bus connection (using python-can)
        self.can0 = CAN()
        print('CAN (python-can, socketcan, Jetson) initialized')
        
    def send_actuation(self, actuation):
        # Convert actuation data to a CAN message
        # ... (implementation specific to CAN bus protocol)
        self.can_bus.send(can_message)

    def receive_state(self, state):
        # Listen for CAN messages containing sensor data
        # ... (implementation specific to CAN bus protocol)
        return sensor_data

class CanKvaser(CarCommunicator):
    def __init__(self, device):
        from can_utils.can_kvaser import CanKvaser
        self.device = device  # Initialize Kvaser device connection
        self.can_receive = CanKvaser()
        self.can_send = CanKvaser()
        self.can_queue = multiprocessing.Queue(maxsize=1) #block=True, timeout=None TODO probably bad parameters, increase maxsize etc.
        
        self.can_send_worker = multiprocessing.Process(target=self._can_send_thread, args=(), daemon=False)
        self.can_send_worker.start()
        
        print('CAN (Kvaser) initialized')

    def send_actuation(self, actuation):
        # Convert actuation data to a CAN message
        # ... (implementation specific to Kvaser CAN bus protocol)
        kvaser_send_message(actuation)

    def receive_state(self, state):
        # Listen for CAN messages containing sensor data
        # ... (implementation specific to Kvaser CAN bus protocol)
        return kvaser_receive_message()

class SimulatorComm(CarCommunicator):
    def __init__(self):
        FSDS_LIB_PATH = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python")
        sys.path.insert(0, FSDS_LIB_PATH)
        print(f"Simulator Comm initialized with FSDS_LIB_PATH: {FSDS_LIB_PATH}")
        import fsds
        self.sim_client2 = fsds.FSDSClient() # To control the car
        # Check network connection, exit if not connected
        self.sim_client2.confirmConnection()
        # Control the Car
        self.sim_client2.enableApiControl(True) # Disconnects mouse control, only API with this code
        self.simulator_car_controls = fsds.CarControls()
        
    def receive_state(self, state):
        """Receives sensor data (speed, RPM, etc.) from the simulator.
        """
        # Read speed from simulator
        sim_state = self.sim_client2.getCarState()
        state['speed'] = sim_state.speed
        
    def send_actuation(self, actuation):
        pass