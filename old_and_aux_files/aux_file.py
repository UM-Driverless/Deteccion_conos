import glob
import cv2

# filenames = glob.glob("/home/shernandez/PycharmProjects/UMotorsport/plt_images/*.png")
# filenames.sort()
# images = [cv2.imread(img) for img in filenames]
# out_image = cv2.VideoWriter('/home/shernandez/PycharmProjects/UMotorsport/plt_image.avi',
#                                 cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (images[0].shape[1], images[0].shape[0]))
#
# for img in images:
#     out_image.write(img)
#
# out_image.release()

#####################################################################################3
cap1 = cv2.VideoCapture('/home/shernandez/PycharmProjects/UMotorsport/image.avi')
cap2 = cv2.VideoCapture('/home/shernandez/PycharmProjects/UMotorsport/plt_image.avi')

out = cv2.VideoWriter('/home/shernandez/PycharmProjects/UMotorsport/composed.avi',
                                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (1033, 579))

# Read until video is completed
while(cap1.isOpened() and cap2.isOpened()):
  # Capture frame-by-frame
  ret1, frame1 = cap1.read()
  ret2, frame2 = cap2.read()
  if ret1 and ret2:

    # Display the resulting frame
    frame1 = cv2.resize(frame1, (1033, 579), interpolation=cv2.INTER_AREA)
    frame2 = cv2.resize(frame2, (282, 206), interpolation=cv2.INTER_AREA)

    frame1[10:181, 10:252] = frame2[10:-25, 30:-10]
    cv2.imshow('Frame1', frame1)
    # cv2.imshow('Frame2', frame2)
    out.write(frame1)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break

out.release()
# When everything done, release the video capture object
cap1.release()
cap2.release()

# Closes all the frames
cv2.destroyAllWindows()