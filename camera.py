import cv2,tensorflow as tf,numpy as np,matplotlib.pyplot as plt

from PIL import Image
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.prediction="happy"

    def __del__(self):
        self.video.release()        

    def get_frame(self):
        ret, frame = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', frame)       
        return jpeg.tobytes()
    def getEmotion(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        ret, frame1 = self.video.read()
        ret,jpeg_1=cv2.imencode('.jpg',frame1)
        model=tf.keras.models.load_model('my_model.h5')
        jpeg_1=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(jpeg_1, 1.3, 5)
        for (x,y,w,h) in faces:
            jpeg_1= jpeg_1[y:y+h, x:x+w]
        cv2.imwrite('photo.jpg',jpeg_1)
        jpeg_1=cv2.resize(jpeg_1,(48,48))
        jpeg_1=np.array(jpeg_1)
        jpeg_1=np.expand_dims(jpeg_1, axis=0)
        result=model.predict(jpeg_1)
        max=0
        maxpos=0
        for i in range(7):
            if(result[0][i]>max):
                max=result[0][i]
                maxpos=i
        dict={0:'anger',1:'disgust',2:'fear',3:'happiness',4:'neutral',5:'sadness',6:'surprise'}
        self.prediction=dict.get(maxpos)  
        return self.prediction

    def get_frame1(self):
        jpeg=cv2.imread('photo.jpg',0)
        ret,jpeg=cv2.imencode('.jpg',jpeg)
        return jpeg.tobytes()

