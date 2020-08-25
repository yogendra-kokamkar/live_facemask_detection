import numpy as np
import keras
import os
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from keras.models import load_model

loaded_model = tf.compat.v1.keras.experimental.load_from_saved_model('mask_detction_model', custom_objects={'KerasLayer':hub.KerasLayer})
print(loaded_model.get_config())
print(loaded_model.summary())

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

source=cv2.VideoCapture(0)

labels_dict={0:'with_mask',1:'without_mask'}
color_dict={0:(0,255,0),1:(0,0,255)}

while(True):

    ret,img=source.read()
    faces=face_clsfr.detectMultiScale(img,1.3,5)  

    for x,y,w,h in faces:
    
        face_img=img[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(299,299))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,299,299,3))
        result=loaded_model.predict(reshaped)

        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(
          img, labels_dict[label], 
          (x, y-10),
          cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()