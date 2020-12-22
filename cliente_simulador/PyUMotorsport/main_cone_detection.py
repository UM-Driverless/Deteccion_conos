import time, os, cv2
import numpy as np
from PIL import Image
from myTools.my_client import bind2server
from cone_detection.detection_utils import ConeDetector


if __name__ == '__main__':
    # Link is established with the server
    mySocket = bind2server()

    # Message format [imageWidth, imageHeigth, numberofCameras, decimalAccuracy, throttle, speed, steer, brake, image]
    index = [4, 8, 12, 16, 20, 24, 28, 29]
    checkpoint_path = './cone_detection/saved_models/ResNet50_640x640_synt_2'
    detector = ConeDetector(checkpoint_path)
    # Image processing
    try:
        mySocket.sendall("ready".encode())
        while True:
            msg = mySocket.recv(8)
            imageWidth = int.from_bytes(msg[:index[0]], byteorder='little', signed=False)
            imageHeigth = int.from_bytes(msg[index[0]:index[1]], byteorder='little', signed=False)
            sum = 8
            tam = imageWidth * imageHeigth * 3 + 29
            is_brake = False
            while sum < tam:
                data = mySocket.recv(tam)
                sum += len(data)
                # msg.extend(data)
                msg = b"".join([msg, data])

            # Message content
            imageWidth = int.from_bytes(msg[:index[0]], byteorder='little', signed=False)
            imageHeigth = int.from_bytes(msg[index[0]:index[1]], byteorder='little', signed=False)
            numberOfCameras = int.from_bytes(msg[index[1]:index[2]], byteorder='little', signed=False)
            decimalAccuracy = int.from_bytes(msg[index[2]:index[3]], byteorder='little', signed=False)
            throttle = int.from_bytes(msg[index[3]: index[4]], byteorder='little', signed=True)/decimalAccuracy
            speed = int.from_bytes(msg[index[4]:index[5]], byteorder='little', signed=False)
            steer = int.from_bytes(msg[index[5]:index[6]], byteorder='little', signed=True)/decimalAccuracy
            brake = int.from_bytes(msg[index[6]:index[7]], byteorder='little', signed=False)

            imageSize = (imageWidth * imageHeigth * 3) // numberOfCameras
            print(f'imageWidth: {imageWidth}, imageHeigth: {imageHeigth}, imageSize: {imageSize}, numberOfCameras: {numberOfCameras}')
            print(f'decimalAccuracy: {decimalAccuracy}, throttle: {throttle}, speed: {speed}, steer: {steer}, brake: {brake}')

            # Paint images
            image = []
            _index = index[7]
            # aaa = int.from_bytes(msg[_index:imageSize + _index])
            for i in range(numberOfCameras):
                image.append(Image.frombytes("RGB", (imageWidth, imageHeigth//numberOfCameras), msg[_index:imageSize + _index]).transpose(
                    method=Image.FLIP_TOP_BOTTOM))
                # cv2.imshow('output'+str(i), cv2.cvtColor(np.array(image[i]), cv2.COLOR_RGB2BGR))
                _index += imageSize

            detector.detect_in_image(np.array(image[0]), plot=True, min_score_thresh=0.3, real_time=True, im_name='Left')
            # detector.detect_in_image(np.array(image[1]), plot=True, min_score_thresh=0.3, real_time=True, im_name='Rigth')


            # Sending the message
            mySocket.sendall("go".encode())  # <- Send a correct action.

            if cv2.waitKey(1) == ord('q'):
                break
        mySocket.close()
    finally:
        mySocket.close()

