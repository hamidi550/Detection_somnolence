import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

# ceci est utilisé pour obtenir un bip sonore (lorsque la personne ferme les yeux pendant plus de 10 secondes)
mixer.init()
sound = mixer.Sound('alarm.wav')
# Ces fichies xml sont utilisés pour détecter le visage, l'oeil gauche et l'oeil droite:
face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')


lbl=['Close','Open']
# Charger le modèle, que nous avons créé:
model = load_model('models/cnncat2.h5')
path = os.getcwd()

# pour capturer chaque image
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

#déclarer des variables
count=0
time=0
thicc=2
rpred=[99]
lpred=[99]

while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2]

#convertir l'image capturée en couleur grise :
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# effectuer la détection (cela renverra les coordonnées x, y, la hauteur, la largeur de l'objet des boîtes aux limites)
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

#itérer sur les faces et tracer des cadres de délimitation pour chaque face
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

#itérer sur l'oeil droit
    for (x,y,w,h) in right_eye:
#retirez l'image de l'œil droit du cadre :
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict_classes(r_eye)
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

#itérer sur l'œil gauche :
    for (x,y,w,h) in left_eye:
#retirez l'image de l'œil gauche du cadre :
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict_classes(l_eye)
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break

    if(rpred[0]==0 and lpred[0]==0):
        time=time+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        time=time-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(time<0):
        time=0   
    cv2.putText(frame,'time:'+str(time),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(time>10):
 #personne se sent étourdie nous allons alerter :
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
            
        except:  # isplaying = False
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc)
    cv2.imshow('Detection de la Somnolence au Volant_PFE_FSR_IDDL_2022',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
