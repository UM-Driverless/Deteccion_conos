import os, sys, cv2, time, serial
import numpy as np
import multiprocessing
from abc import ABC, abstractmethod

class CarCommunicator(ABC):
    def __init__(self):
        # FSDS_LIB_PATH = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python") # os.getcwd()
        self.SRC_DIR = os.path.dirname(os.path.abspath(__file__))
        self.ROOT_DIR = os.path.dirname(self.SRC_DIR)

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
        elif mode == 'sim':
            return SimulatorComm()
        else:
            raise ValueError(f'Unknown comm mode: {mode}')
        
class NullCarCommunicator(CarCommunicator):
    def send_actuation(self, actuation):
        # Do nothing (no actuation sent)
        pass

    def receive_state(self, state):
        # Return empty state data (no data received)
        return {}

class SimulatorComm(CarCommunicator):
    def __init__(self):
        super().__init__()
        FSDS_LIB_PATH = os.path.join(os.path.dirname(self.ROOT_DIR), "Formula-Student-Driverless-Simulator", "python")
        sys.path.insert(0, FSDS_LIB_PATH)
        print(f'FSDS simulator path: {FSDS_LIB_PATH}')
        global fsds
        import fsds
        
        self.client = fsds.FSDSClient() # To control the car
        # Check network connection, exit if not connected
        self.client.confirmConnection()
        # Control the Car
        self.client.enableApiControl(True) # Disconnects mouse control, only API with this code
        self.controls = fsds.CarControls()
        
    def receive_state(self, state):
        """Receives sensor data (speed, RPM, etc.) from the simulator.
        """
        # Read speed from simulator
        sim_state = self.client.getCarState()
        state['speed'] = sim_state.speed
        
    def send_actuation(self, actuation):
        # Send to Simulator
        self.controls.steering = -actuation['steer'] # + rotation is left for us, right for simulator
        self.controls.throttle = actuation['acc']
        self.controls.brake = actuation['brake']

        self.client.setCarControls(self.controls)

class SerialCommunicator(CarCommunicator):
    """
    Communicator for the testing Robot, that will connect to the Arduino via serial port (USB-B). The Arduino has the shield with the motor drivers.
    TODO IMPLEMENT THIS FOR ROBOT
    """
    def __init__(self, port):
        # self.serial_port = serial.Serial(port)  # Initialize serial connection
        self.arduino = serial.Serial(port='COM6', baudrate=9600, timeout=.1)
        
    # def test(self):
    #     self.arduino.write(bytes('1', 'utf-8'))
    #     self.arduino.write(bytes('2', 'utf-8'))
    #     self.arduino.write(bytes('3', 'utf-8'))
    #     self.arduino.write(bytes('1', 'utf-8'))
    #     self.arduino.write(bytes('2', 'utf-8'))
    #     self.arduino.write(bytes('3', 'utf-8'))

    def send_actuation(self, actuation):
        # The arduino needs to receive the angle in degrees, from 0 to 180deg
        # Map from (-90, 90) to (0, 180), and convert to int
        
        angle = int((actuation['steer'] + 1) / 2 * 90)
        data = bytes(str(angle), 'utf-8') # + rotation is left for us, right for simulator
        self.arduino.write(data)

    def receive_state(self, state):
        # Read data from serial port and parse it into sensor data
        # ... (implementation specific to serial protocol)
        return 0

class CanJetson(CarCommunicator):
    def __init__(self, can_bus):
        from can_utils.can_utils import CAN
        self.can_bus = can_bus  # Initialize CAN bus connection (using python-can)
        self.can0 = CAN()
        print('CAN (python-can, socketcan, Jetson) initialized')
        
    def send_actuation(self, actuation):
        # Convert actuation data to a CAN message
        # ... (implementation specific to CAN bus protocol)
        self.can0.send_action_msg(actuation)

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