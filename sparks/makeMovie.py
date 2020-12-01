import cv2
import os
import sys

image_folder = sys.argv[1]
video_name = f'{image_folder}.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".tif")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 30, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
