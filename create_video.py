import cv2
import os

image_folder = '/media/archivos/UMotorsport/ImagenesDataset/ImagenesSinEtiquetar1/aux/'
video_name = '/media/archivos/UMotorsport/video_28_01_2018__16_59_1-622.png.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort(key= lambda x:int(x.split('-')[1].split('.')[0]))
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape


fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_name, fourcc, 2, (width, height))

name = 'video_{:0>4d}.jpg'
# images = [name.format(i) for i in range(0, 661)]
for image in images:
    image = cv2.imread(os.path.join(image_folder, image))
    cv2.imshow("im", image)
    cv2.waitKey(1)
    video.write(image)

cv2.destroyAllWindows()
video.release()