import cv2
import os
import numpy as np
import import_ipynb 
import faceRecognition as fr

test_img = cv2.imread('kangnaranaut001.jpg')
faces_detected,gray_img = fr.faceDetection(test_img)
print("Face Detected:",faces_detected)
#faces,faceID=fr.lebels_for_tranning_data('trannning-data') # To Train the Model
#face_recognizer=fr.train_classifier(faces,faceID)
#face_recognizer.save("tranningData.yml")#Onece Trainning complete Save the model in tranningData.yml
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('tranningData.yml')#from The model data predicting the face

name={0:"Priyanka",1:"Kangana"}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print("Confidence",confidence)
    print("Label:",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    fr.put_text(test_img,predicted_name,x,y)
    

resized_img = cv2.resize(test_img,(800,500))
cv2.imshow("Test Picture",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows

