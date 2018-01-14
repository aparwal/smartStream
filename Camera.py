import time
import cv2
import numpy as np
# import urllib2
# import base64

url='http://130.215.122.188:8080/video'
cascPath = "haarcascade_frontalface_default.xml"

def motion_scorer(frame,oldFrame,motion_scale=1):
    if (oldFrame) is not None:
        frameDelta = cv2.absdiff(frame, oldFrame)
        # cv2.imshow("frameDelta", frameDelta)
        change = np.sum(frameDelta ** 2)
        return change*motion_scale/(frame.shape[0]*frame.shape[1])
    else:
        return 0

def face_detector(gray,cascPath):

    faceCascade = cv2.CascadeClassifier(cascPath)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces

def get_face_metrics(frame,cascPath= cascPath):
    faces=face_detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),cascPath)
    # get ratios of face box to the screen size
    face_ratio=0
    for (x, y, w, h) in faces:
        face_ratio+=w*h/(frame.shape[0]*frame.shape[1])
    return len(faces),face_ratio

class Camera(object):

    def __init__(self, camera=0):
        self.cam = cv2.VideoCapture(camera)
        print("Camera Initiated at ",camera)
        if not self.cam:
            raise Exception("Camera not accessible at ",str(camera))

        self.shape = self.get_frame().shape

        self.gray = cv2.cvtColor(self.get_frame(), cv2.COLOR_BGR2GRAY)

    def shut_down(self):
        self.cam.release()

    def get_frame(self):
        _, frame = self.cam.read()
        return frame

    def get_faces(self,cascPath = cascPath):
        return face_detector(cv2.cvtColor(self.get_frame(), cv2.COLOR_BGR2GRAY),cascPath)

    def get_face_metrics(self,cascPath= cascPath):
        faces=self.get_faces()
        # get ratios of face box to the screen size
        face_ratio=0
        for (x, y, w, h) in faces:
            face_ratio+=w*h/(self.shape[0]*self.shape[1])
        return len(faces),face_ratio


'''# depricated
class ipCamera(object):

    def __init__(self, url, user=None, password=None):
        self.url = url
        # auth_encoded = base64.encodestring('%s:%s' % (user, password))[:-1]

        self.req = urllib.Request(self.url)
        # self.req.add_header('Authorization', 'Basic %s' % auth_encoded)

    def get_frame(self):
        response = urllib2.urlopen(self.req)
        img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, 1)
        return frame
'''




if __name__ == '__main__':
    
    # Initiate capture objects
    # phone=Camera(url)
    webcam=Camera(0)

    while True:
        img=webcam.get_frame()


        # Draw a rectangle around the faces
        for (x, y, w, h) in webcam.get_faces():
          cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
        # Display the resulting frame
        # cv2.imshow('phone', phone.get_frame())

        cv2.imshow('webcam',img)
        one,two=webcam.get_face_metrics()
        print(one,two)
        k=cv2.waitKey(1)

        if k == 27:
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()