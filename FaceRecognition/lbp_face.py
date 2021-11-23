import cv2
from faces import load_face_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import imutils
import numpy
import os

inputPath = "/Users/shireen/Documents/opencv/CVProjects/FaceRecognition/photos"
prototxt = "/Users/shireen/Documents/opencv/CVProjects/FaceRecognition/deploy.prototxt.txt"
caffemodel = "/Users/shireen/Documents/opencv/CVProjects/FaceRecognition/res10_300x300_ssd_iter_140000.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototxt,caffemodel)
(faces,labels) = load_face_dataset(inputPath,net,minConfidence=0.5,minSamples=20)

le = LabelEncoder()
labels = le.fit_transform(labels)

(X_train,X_test,y_train,y_test) = train_test_split(faces,labels,test_size=0.25,stratify=labels,random_state=42)

recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2,neighbors=16,grid_x=8,grid_y=8)

recognizer.train(X_train,y_train)

predictions = []
confidence = []

for i in range(0,len(X_test)):
	(prediction,conf) = recognizer.predict(X_test[i])
	predictions.append(prediction)
	confidence.append(conf)

print(classification_report(y_test,predictions,target_names=le.classes_))

