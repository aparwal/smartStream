import cv2
import sys
import numpy as np
from Camera import *

# device locations
url='http://130.215.122.188:8080/video'
locations=[0,url]

# config parameters
fps=2
weight=[1,1,.1,1]
motion_scale=[1,1]
motion_score=[0,0]

# initiating variables
devices=[]
prev_frame=[0,0]
score=[0,0]
framenos=0
frame=[0,0]

# scoring function
def cal_score(partialscores):
	return np.sum(np.array(weight)*np.array(partialscores))

def display(devices,scores):
	selected=devices[np.argmax(score)].get_frame()
	cv2.imshow('Video', selected)
	k=cv2.waitKey(1)

	if k == 27:
		return True
	else: 
		return False

# Initiate and get first frames
for i in range(len(locations)):
	devices.append(Camera(locations[i]))
	prev_frame[i] = cv2.cvtColor(devices[i].get_frame(), cv2.COLOR_BGR2GRAY)
	prev_frame[i] = cv2.GaussianBlur(prev_frame[i], (21, 21), 0)

# Let it rip
while True:

	framenos+=1

	# Draw a rectangle around the faces
	# for (x, y, w, h) in device.get_faces():
	# 	cv2.rectangle(device.get_frame(), (x, y), (x+w, y+h), (0, 255, 0), 2)

	#process on the nth frame
	if framenos%fps == 0:
		
		framenos = 0


		# for all devices
		for i in range(len(devices)):

			frame[i]=devices[i].get_frame()

			face_nos,face_ratio=get_face_metrics(frame[i])	        

			# get motion score
			img = cv2.GaussianBlur(cv2.cvtColor(frame[i], cv2.COLOR_BGR2GRAY), (21, 21), 0)
			motion_score = motion_scorer(img,prev_frame[i],motion_scale[i])
			prev_frame[i] = img

			speech=0

			partialscores=[face_nos,0,motion_score,speech]
			# print(i,' : ',partialscores)
			score[i]=cal_score(partialscores)

		print ("\rfirst: {:.2f}\t second: {:.2f}".format(score[0],score[1]),end='')

		# print("\rpeople = {}   prominance = {:.2f}   motion = {:04.1f}   speech = 0".format(face_nos,face_ratio,motion_score) ,end="")

	# Display the resulting frame
	if display(devices,score):
		break;
	# selected=devices[np.argmax(score)].get_frame()
	# cv2.imshow('Video', selected)
	# k=cv2.waitKey(1)

	# if k == 27:
	# 	break

# When everything is done, release the capture
for device in devices:
	device.shut_down()
print()
cv2.destroyAllWindows()