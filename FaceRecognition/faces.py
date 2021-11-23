import cv2
import numpy as np
import os

def detect_faces(net,image,minConfidence=0.5):
	(h,w) = image.shape[:2]

	blob = cv2.dnn.blobFromImage(image,1,(300,300),(104,177,123))
	net.setInput(blob)
	detections = net.forward()
	boxes = []

	for i in range(0,detections.shape[2]):

		confidence = detections[0,0,i,2]

		if confidence > minConfidence:
			box = detections[0,0,i,3:7] * np.array([w,h,w,h])
			(startX,startY,endX,endY) = box.astype("int")

			boxes.append((startX,startY,endX,endY))


	return boxes


def load_face_dataset(inputPath,net,minConfidence=0.5,minSamples=15):
	imagePaths = []
	names = []
	faces = []
	labels = []

	os.chdir(inputPath)
	for dir in os.listdir():
		if dir == '.DS_Store':
			continue
		dirpath = f"{inputPath}/{dir}"
		name = dir.split('-')[0] + " "+ dir.split('-')[1]

		os.chdir(dirpath)
		for image in os.listdir():
			imagePath = f"{dirpath}/{image}"
			imagePaths.append(imagePath)
			names.append(name)

	(names,counts) = np.unique(names,return_counts=True)
	names = names.tolist()


	for imagePath in imagePaths:
		if '.DS_Store' in imagePath:
			continue
		image = cv2.imread(imagePath)
		name = imagePath.split('/')[8]
		name = name.split('-')[0] + " " + name.split('-')[1]

		if counts[names.index(name)] < minSamples:
			continue

		boxes = detect_faces(net,image,minConfidence)

		for (startX,startY,endX,endY) in boxes:
			faceROI = image[startY:endY,startX:endX]
			faceROI = cv2.resize(faceROI,(47,62))
			faceROI = cv2.cvtColor(faceROI,cv2.COLOR_BGR2GRAY)


			faces.append(faceROI)
			labels.append(name)

	faces = np.array(faces)
	labels = np.array(labels)

	return (faces,labels)















