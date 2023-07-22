'''
Script dedicated to the recording of images received by the server
'''
import cv2, os, shutil
import numpy as np
from PIL import Image
from can.my_client import bind2server

def recordImages(images, count, path1, path2, format="JPEG"):
    '''Save the images in the specified path and format, with an index by name'''
    images[0].save(path1 + '/' + str(count) + ".jpg", format)
    if path2 != None:
        image[1].save(path2 + '/' + str(count) + ".jpg", format)

def createDirectory(directory, stereo=False):
    '''Creates a directory in the specified path and overwrites if it exists'''
    if (os.path.exists(directory)):
        shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(directory)
    if stereo:
        path1 = directory+"/right_camera"
        path2 = directory+"/left_camera"
        os.makedirs(path1)
        os.makedirs(path2)
    else:
        return directory, None
    return path1, path2

if __name__ == '__main__':
    # Link is established with the server
    mySocket = bind2server()

    # Parameters
    imageCounter = 0
    path1, path2 = createDirectory("../data/skidpad", stereo=False)
    msgFormat = [4, 8, 12, 16, 20, 24, 28, 29] # Message format [imageWidth, imageHeigth, numberofCameras, decimalAccuracy, throttle, speed, steer, brake, image]

    # Image processing
    try:
        mySocket.sendall("ready".encode())
        while True:
            msg = mySocket.recv(8388608)

            # Message content
            imageWidth = int.from_bytes(msg[:msgFormat[0]], byteorder='little', signed=False)
            imageHeigth = int.from_bytes(msg[msgFormat[0]:msgFormat[1]], byteorder='little', signed=False)
            numberOfCameras = int.from_bytes(msg[msgFormat[1]:msgFormat[2]], byteorder='little', signed=False)
            decimalAccuracy = int.from_bytes(msg[msgFormat[2]:msgFormat[3]], byteorder='little', signed=False)
            throttle = int.from_bytes(msg[msgFormat[3]: msgFormat[4]], byteorder='little', signed=True) / decimalAccuracy
            speed = int.from_bytes(msg[msgFormat[4]:msgFormat[5]], byteorder='little', signed=False)
            steer = int.from_bytes(msg[msgFormat[5]:msgFormat[6]], byteorder='little', signed=True) / decimalAccuracy
            brake = int.from_bytes(msg[msgFormat[6]:msgFormat[7]], byteorder='little', signed=False)

            imageSize = (imageWidth * imageHeigth * 3) // numberOfCameras
            print(
                f'imageWidth: {imageWidth}, imageHeigth: {imageHeigth}, imageSize: {imageSize}, numberOfCameras: {numberOfCameras}')
            print(
                f'decimalAccuracy: {decimalAccuracy}, throttle: {throttle}, speed: {speed}, steer: {steer}, brake: {brake}')

            # Get images
            image = [] # list of cameras
            _msgFormat = msgFormat[7]
            for i in range(numberOfCameras):
                image.append(
                    Image.frombytes("RGB", (imageWidth, imageHeigth // numberOfCameras), msg[_msgFormat:imageSize + _msgFormat]).transpose(
                        method=Image.FLIP_TOP_BOTTOM))
                _msgFormat += imageSize

            # Record images
            recordImages(image, imageCounter, path1, path2, format="JPEG")
            imageCounter += 1

            # Sending the message
            mySocket.sendall("go".encode())
            if cv2.waitKey(1) == ord('q'):
                break
        mySocket.close()
    finally:
        mySocket.close()