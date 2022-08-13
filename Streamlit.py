import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf

st.title("Welcome to Song Recommendation by Emotion Detection")
picture=st.camera_input('Capture Face for Recognition')
if picture is not None:
    st.image(picture)
    img=Image.open(picture)
    picture=np.array(img)
    # picture
    # cv2.imwrite('photo.jpg',picture)
    # picture=cv2.imread('photo.jpg',0)
    picture=cv2.resize(picture,(48,48))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    ret,jpeg_1=cv2.imencode('.jpg',picture)
    model=tf.keras.models.load_model('my_model.h5')
    faces = face_cascade.detectMultiScale(jpeg_1, 1.3, 5)
    for (x,y,w,h) in faces:
            jpeg_1= jpeg_1[y:y+h, x:x+w]
    cv2.imwrite('photo.jpg',jpeg_1)
    jpeg_1=np.array(jpeg_1)
    jpeg_1=cv2.resize(jpeg_1,(48,48))
    jpeg_1=np.expand_dims(jpeg_1, axis=0)
    result=model.predict(jpeg_1)
    max=0
    maxpos=0
    for i in range(7):
        if(result[0][i]>max):
            max=result[0][i]
            maxpos=i
    dict={0:'anger',1:'disgust',2:'fear',3:'happiness',4:'neutral',5:'sadness',6:'surprise'}
    prediction=dict.get(maxpos)
    st.write(prediction)
    dict1={0:'https://open.spotify.com/playlist/71Xpaq3Hbpxz6w9yDmIsaH',
           1:'https://open.spotify.com/playlist/3qgzMg4m5tvf16PzlPgGa9',
           2:'https://open.spotify.com/playlist/7rzS9iLiqjy65AsZd9qinf',
           3:'https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC',
           4:'https://open.spotify.com/playlist/37i9dQZF1DWTC99MCpbjP8',
           5:'https://open.spotify.com/playlist/37i9dQZF1DX7qK8ma5wgG1',
           6:'https://open.spotify.com/playlist/0X0ZZTJ6z2yxX5Uu7R7j3G'
          }
    song_link=dict1.get(maxpos)
    st.write(song_link)
