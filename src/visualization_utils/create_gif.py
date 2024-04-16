import imageio
import os

image_folder = '/media/archivos/UMotorsport/aux/'
gif_name = '/media/archivos/UMotorsport/video_21_11_21_16_59_1_yolo.gif'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
# images.sort(key= lambda x:int(x.split('-')[1].split('.')[0]))
images.sort()

with imageio.get_writer(gif_name, mode='I') as writer:
    for filename in images:
        image = imageio.imread(image_folder+filename)
        writer.append_data(image)