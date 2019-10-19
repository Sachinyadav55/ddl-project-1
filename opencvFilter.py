#!/usr/bin/env python
# coding: utf-8

#Install imtils and openCV if not already installed
#!pip install imutils
#!pip install opencv-python

import cv2
from imutils.video import VideoStream
import imutils
import time, sys, argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=300, help="minimum area size")
args = vars(ap.parse_args())

if args.get("video", None) is None:
	print ('Provide Video Path using -v flag')
	sys.exit()

else:
	vs = cv2.VideoCapture(args["video"])
	length = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
	print ('Number of frames in video: '+ str(length))

# initialize the first frame in the video stream
firstFrame = None
frameID = 0
xl = [['frameNumber', 'include']]
# loop over the frames of the video
while True:
	frame = vs.read()
	frame = frame if args.get("video", None) is None else frame[1]
	
	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if frame is None:
		break

	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	frameID += 1
	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		xl.append([frameID, 0])
		continue
		# compute the absolute difference between the current frame and
	# first frame
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	include = False
	# loop over the contours
	for c in cnts:
		if cv2.contourArea(c) > args["min_area"]:
			include = True
			break

	if include: xl.append([frameID, 1])
	else: xl.append([frameID, 0])

df = pd.DataFrame(xl)
df.to_csv(args["video"].split(".")[0] + '.csv', index=False, header=None)