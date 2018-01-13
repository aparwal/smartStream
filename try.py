import cv2
import sys
import numpy as np

def motion_scorer(frame,oldFrame):
    if (oldFrame) is not None:
        frameDelta = cv2.absdiff(frame, oldFrame)
        cv2.imshow("frameDelta", frameDelta)
        change = np.sum(frameDelta ** 2)
        return change/1000000
    else:
        return 0

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
framenos=0
fps=2
ret, prev_frame = video_capture.read()
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    framenos+=1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    #process on the nth frame
    if framenos%fps == 0:
        
        framenos = 0

        # get num of faces
        face_nos = len(faces)

        # get ratios of face box to the screen size
        face_ratio=0
        for (x, y, w, h) in faces:
            face_ratio+=w*h/(frame.shape[0]*frame.shape[1])
        

        # get motion score
        img = cv2.GaussianBlur(gray, (21, 21), 0)
        motion_score = motion_scorer(img,prev_frame)
        prev_frame = img

        print("\rpeople = {}   prominance = {:.2f}   motion = {:04.1f}   speech = 0".format(face_nos,face_ratio,motion_score) ,end="")


    # Display the resulting frame
    cv2.imshow('Video', frame)
    k=cv2.waitKey(1)

    if k == 27:
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()