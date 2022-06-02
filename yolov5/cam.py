import cv2
import torch
import models
import utils
import numpy as np
import time

np.random.seed(42)
colors = np.random.randint(0, 255, (6, 3), dtype=np.uint8)

capture = cv2.VideoCapture(0)
model = torch.hub.load('.', 'custom', path='weights/yolov5s-cones-mixed-classes/weights/best.pt', source='local', force_reload=True)
while (capture.isOpened()):
    ret, frame = capture.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frameRGB)
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        x2 = int(row['xmax'])
        y1 = int(row['ymin'])
        y2 = int(row['ymax'])
        label = '{} {:.1f}'.format(row['name'],row['confidence'])
        color = [int(c) for c in colors[int(row['class'])]]

        # For bounding box
        img = cv2.rectangle(frame, (x1, y1), (x2, y2), (color[0],color[1],color[2]), 2)

        # For the text background
        # Finds space required by the text so that we can put a background with that amount of width.
        (w, h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # Prints the text.
        img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (color[0],color[1],color[2]), -1)
        #img = cv2.putText(img, label, (x1, y1 - 5),
        #                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

        # For printing text
        img = cv2.putText(img, label, (x1, y1),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    #results.show()
    cv2.imshow('webCam',frame)
    if (cv2.waitKey(1) == ord('s')):
        break

capture.release()
cv2.destroyAllWindows()