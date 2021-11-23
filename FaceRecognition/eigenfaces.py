from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.exposure import rescale_intensity
from faces import load_face_dataset
from imutils import build_montages
import numpy as np
import imutils
import time
import cv2
import os

inputPath = "/Users/shireen/Documents/opencv/CVProjects/FaceRecognition/photos"
prototxt = "/Users/shireen/Documents/opencv/CVProjects/FaceRecognition/deploy.prototxt.txt"
caffemodel = "/Users/shireen/Documents/opencv/CVProjects/FaceRecognition/res10_300x300_ssd_iter_140000.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototxt,caffemodel)

minConfidence = 0.5
num_principal_components = 50
visualize = 1

(faces,labels) = load_face_dataset(inputPath,net,minConfidence,minSamples=20)

pcaFaces = np.array([f.flatten() for f in faces])

le = LabelEncoder()
labels = le.fit_transform(labels)

(origTrain,origTest,X_train,X_test,y_train,y_test) = train_test_split(faces,pcaFaces,labels,test_size=0.25,stratify=labels,random_state=42)

pca = PCA(
	svd_solver="randomized",
	n_components=num_principal_components,
	whiten=True)

X_train = pca.fit_transform(X_train)
if visualize > 0:
	images = []

	for(i,component) in enumerate(pca.components_[:16]):
		component = component.reshape((62,47))
		component = rescale_intensity(component,out_range=(0,255))
		component = np.dstack([component.astype("uint8")] * 3)

	montage = build_montages(images,(47,62),(4,4))[0]

	mean = pca.mean_.reshape((62,47))
	mean = rescale_intensity(mean,out_range=(0,255)).astype("uint8")


model = SVC(kernel="rbf",C=10.0,gamma=0.001,random_state=42)
model.fit(X_train,y_train)


predictions = model.predict(pca.transform(X_test))
print(classification_report(y_test,predictions,target_names=le.classes_))






