# READING VIDOES

# A video can be read either by using the feed from a camera connected to a computer or by reading a video file.
# Displaying a video is done frame by frame. 
# A frame of a video is simply an image and we display each frame the same way we display images.

import cv2 
 
# Create a video capture object, in this case we are reading the video from a file
vid_capture = cv2.VideoCapture('road.mp4')
# vid_capture = cv2.VideoCapture(0)
if (vid_capture.isOpened() == False):
  print("Error opening the video file")
# Read fps and frame count
else:
  # Get frame rate information
  # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
  fps = vid_capture.get(5)
  print('Frames per second : ', fps,'FPS')
 
  # Get frame count
  # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
  frame_count = vid_capture.get(7)
  print('Frame count : ', frame_count)
 
while(vid_capture.isOpened()):
  # vid_capture.read() methods returns a tuple, first element is a bool 
  # and the second is frame
  ret, frame = vid_capture.read()
  if ret == True:
    cv2.imshow('Frame',frame)
    # 20 is in milliseconds, try to increase the value, say 50 and observe
    key = cv2.waitKey(30)
     
    if key == ord('q'):
      break
  else:
    break
 
# Release the video capture object
vid_capture.release()
cv2.destroyAllWindows()