'''
Plugin to establish connection and exchange text messages between two platforms. Producing
this communication on the same machine.
'''
from connection_utils.comunication_base import ComunicationInterface
import socket
import cv2
from PIL import Image
import numpy as np


class ConnectionManager(ComunicationInterface):
    def __init__(self, ip=None, port=12345):
        self.mySocket = self._bind2server(ip, port)
        self.msg_size, self.channels, self.parameters_size = 8, 3, 29
        self.mySocket.sendall("ready".encode())
        print("Established connection")

        self.car_data_sim = carr_data_simulation()
        # Message format [imageWidth, imageHeigth, numberofCameras, decimalAccuracy, throttle, speed, steer, brake, image]
        self.msg_index = [4, 8, 12, 16, 20, 24, 28, 29]
        self.imageWidth = None
        self.imageWidth = None

        _ = self.get_data()
        self.send_actions(throttle=0., brake=1., steer=0., clutch=1.0, upgear=0., downgear=0.)

    def _bind2server(self, HOST=None, PORT=12345):
        '''
        Establishes a connection with the server, which is listening on port 12345 of this machine.
        '''
        if HOST is None:
            HOST = socket.gethostname()
            HOST = socket.gethostbyname(HOST)
        print(f'IP: {HOST}, Port: {PORT}')

        # Create a socket (SOCK_STREAM means a TCP socket)
        mySocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        mySocket.connect((HOST, PORT))
        return mySocket

    def get_data(self, params=None, verbose=0):
        # Message reconstruction
        msg = self.mySocket.recv(self.msg_size)
        self.imageWidth = int.from_bytes(msg[:self.msg_index[0]], byteorder='little', signed=False)
        self.imageHeigth = int.from_bytes(msg[self.msg_index[0]:self.msg_index[1]], byteorder='little', signed=False)
        sum = self.msg_size
        tam = self.imageWidth * self.imageHeigth * self.channels + self.parameters_size
        while sum < tam:
            data = self.mySocket.recv(tam)
            sum += len(data)
            msg = b"".join([msg, data])

        # Message content
        imageWidth = int.from_bytes(msg[:self.msg_index[0]], byteorder='little', signed=False)
        imageHeigth = int.from_bytes(msg[self.msg_index[0]:self.msg_index[1]], byteorder='little', signed=False)
        numberOfCameras = int.from_bytes(msg[self.msg_index[1]:self.msg_index[2]], byteorder='little', signed=False)
        decimalAccuracy = int.from_bytes(msg[self.msg_index[2]:self.msg_index[3]], byteorder='little', signed=False)
        throttle = int.from_bytes(msg[self.msg_index[3]: self.msg_index[4]], byteorder='little',
                                  signed=True) / decimalAccuracy
        speed = int.from_bytes(msg[self.msg_index[4]:self.msg_index[5]], byteorder='little', signed=False)
        steer = int.from_bytes(msg[self.msg_index[5]:self.msg_index[6]], byteorder='little',
                               signed=True) / decimalAccuracy
        brake = int.from_bytes(msg[self.msg_index[6]:self.msg_index[7]], byteorder='little', signed=False)

        image = []
        _index = self.msg_index[7]
        imageSize = (imageWidth * imageHeigth * 3) // numberOfCameras

        for i in range(numberOfCameras):
            image.append(Image.frombytes("RGB", (imageWidth, imageHeigth // numberOfCameras),
                                         msg[_index:imageSize + _index]).transpose(method=Image.FLIP_TOP_BOTTOM))

        if verbose == 1:
            print('\r', f'Number_cameras: {numberOfCameras}, throttle: {throttle}, brake: {brake}, steer: {steer}',
                  end='\n')
        elif verbose == 2:
            print(
                '\r',
                f'ImageWidth: {imageWidth}, imageHeigth: {imageHeigth}, imageSize: {imageSize}, number_cameras: {numberOfCameras}, decimalAccuracy: {decimalAccuracy}, throttle: {throttle}, brake: {brake}, steer: {steer}',
                end='\n')
            for i in range(numberOfCameras):
                cv2.imshow('output' + str(i), cv2.cvtColor(np.array(image[i]), cv2.COLOR_RGB2BGR))

        if numberOfCameras < 2:
            image = image[0]

        gear = self.car_data_sim.gear
        rpm = self.car_data_sim.rpm
        self.car_data_sim.speed = speed
        return np.array(image), speed, throttle, steer, brake, gear, rpm

    def send_actions(self, throttle, steer, brake, clutch, upgear, downgear):
        throttle = self.car_data_sim._clutch_response_simulation(clutch, throttle)
        self.car_data_sim.update_car_state(throttle, steer, brake, clutch, upgear, downgear)
        msg = f'{throttle} {brake} {steer}'
        self.mySocket.sendall(msg.encode())  # <- Sends the agent's actions

    def close_connection(self):
        self.mySocket.close()


class carr_data_simulation:
    def __init__(self):
        self.gear = 1
        self.rpm = 1000
        self.speed = 0.

    def update_car_state(self, throttle, steer, brake, clutch, upgear, downgear):
        self.gear += int(upgear) - int(downgear)
        self.rpm = self.calc_rpm(self.gear, self.speed)

    def _clutch_response_simulation(self, clutch, throttle):
        # Simula la relacion entre aceleracion y embrague de forma que se produzca una acceleracion proporcional a la activacion del embrague
        # Simulamos con una función lineal y = mx donde el embrague va a controlar la pendiente.
        m = np.clip(1.5 - clutch, 0., 1.0)
        if m > 0.1:
            throttle = m * throttle
        else:
            throttle = 0.
        return throttle

    def calc_rpm(self, gear, speed):
        if speed <= 4.:
            rpm = 800.
        elif gear == 0:
            rpm = speed * 400
        else:
            rpm = (speed / gear) * 200

        return int(rpm)

# def bind2server(PORT=12345):
#     '''
#     Establishes a connection with the server, which is listening on port 12345 of this machine.
#     '''
#     HOST = socket.gethostname()
#     HOST = socket.gethostbyname(HOST)
#     print(f'IP: {HOST}, Port: {PORT}')
#
#     # Create a socket (SOCK_STREAM means a TCP socket)
#     mySocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     mySocket.connect((HOST, PORT))
#     return mySocket
#
# def send_msg(mySocket, data):
#     mySocket.sendall((data + "\n").encode())
#
# def receive_msg(mySocket):
#     return mySocket.recv(1024)
#
# def print_msg(data, received):
#     print("Sent:     {}".format(data))
#     print("Received: {}".format(received))
#
# # from my_client import bind2server, send_msg, receive_msg, print_msg
# '''
# Script, communication test.
# '''
# if __name__ == '__main__':
#     print("Running client test")
#     mySocket = bind2server()
#     try:
#         data = "hola"
#         send_msg(mySocket, data)
#         serverMsg = receive_msg(mySocket)
#         print_msg(data, serverMsg)
#
#         data = "adios"
#         send_msg(mySocket, data)
#         serverMsg = receive_msg(mySocket)
#         print_msg(data, serverMsg)
#     finally:
#         mySocket.close()
#
