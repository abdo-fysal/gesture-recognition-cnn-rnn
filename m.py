
import cv2
vidcap = cv2.VideoCapture(0)
success,image = vidcap.read()
count = 0
success = True
while count <28:
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  cv2.imwrite("1/New folder/60.%d.jpg" % count, image)     # save frame as JPEG file
  count += 1

