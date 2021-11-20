import cv2
import dlib
import time
import imutils
import numpy as np
from helpers import convert_and_trim_bb

image_path = "/Users/shireen/Documents/opencv/image.jpeg"
haar_path = "/Users/shireen/Documents/opencv/CVProjects/FaceDetectors/haarcascade_frontalface_default.xml"
prototxt_path = "/Users/shireen/Documents/opencv/CVProjects/FaceDetectors/deploy.prototxt.txt"
caffemodel_path = "/Users/shireen/Documents/opencv/CVProjects/FaceDetectors/res10_300x300_ssd_iter_140000.caffemodel"
mmodface_path = "/Users/shireen/Documents/opencv/CVProjects/FaceDetectors/mmod_human_face_detector.dat"

def menu(image_path,haar_path,prototxt_path,caffemodel_path,mmodface_path):
	choice = 1
	while choice != 0:
		image = cv2.imread(image_path)
		print("Make your choice:")
		print("1. Haar Cascades Face Detector")
		print("2. OpenCV SSD Face Detector/Deep Learning Face Detector")
		print("3. Dlib's HOG + Linear SVM Face Detector")
		print("4. Dlib's CNN Face Detector")
		print("0. To exit")
		choice = int(input())

		if choice == 1:
			haar_cascades(image,haar_path)
		elif choice == 2:
			ssd(image,prototxt_path,caffemodel_path)
		elif choice == 3:
			hog(image)
		elif choice == 4:
			cnn(image,mmodface_path)

def haar_cascades(image,haar_path):
	start = time.time()
	haar_detector = cv2.CascadeClassifier(haar_path)
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	rects = haar_detector.detectMultiScale(image,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
	end = time.time()

	for (x,y,w,h) in rects:
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,225,0),2)
	print(f"Detected {len(rects)} faces in image")
	print(f"Executed in {end-start} seconds")
	cv2.imshow("Using Haar Cascades",image)
	cv2.waitKey(0)

def ssd(image,prototxt_path,caffemodel_path):
	start = time.time()
	net = cv2.dnn.readNetFromCaffe(prototxt_path,caffemodel_path)
	(h,w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1.0,(300,300),(104.0,177.0,123.0))
	net.setInput(blob)
	detections = net.forward()
	end = time.time()
	count = 0

	for i in range(0,detections.shape[2]):
		confidence = detections[0,0,i,2]
		if confidence > 0.50:
			count += 1
			box = detections[0,0,i,3:7] * np.array([w,h,w,h])
			(startX,startY,endX,endY) = box.astype("int")
			cv2.rectangle(image,(startX,startY),(endX,endY),(0,255,0),2)

	print(f"Detected {count} faces in image")
	print(f"Executed in {end-start} seconds")
	cv2.imshow("Using OpenCV's SSD",image)
	cv2.waitKey(0)

def hog(image):
	start = time.time()
	hog_detector = dlib.get_frontal_face_detector()
	rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	rects = hog_detector(rgb,1)
	end = time.time()
	boxes = [convert_and_trim_bb(image,r) for r in rects]

	for (x,y,w,h) in boxes:
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

	print(f"Detected {len(boxes)} faces in image")
	print(f"Executed in {end-start} seconds")
	cv2.imshow("Using Dlib's HOG + Linear SVM",image)
	cv2.waitKey(0)

def cnn(image,mmodface_path):
	start = time.time()
	cnn_detector = dlib.cnn_face_detection_model_v1(mmodface_path)
	rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	results = cnn_detector(rgb,1)
	end = time.time()
	boxes = [convert_and_trim_bb(image,r.rect) for r in results]

	for (x,y,w,h) in boxes:
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

	print(f"Detected {len(boxes)} faces in image")
	print(f"Executed in {end-start} seconds")
	cv2.imshow("Using Dlib's CNN",image)
	cv2.waitKey(0)


menu(image_path,haar_path,prototxt_path,caffemodel_path,mmodface_path)










	