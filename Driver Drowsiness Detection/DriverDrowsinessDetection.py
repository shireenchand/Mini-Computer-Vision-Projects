import cv2
import dlib
import numpy as np
import playsound
import imutils
from scipy.spatial import distance as dist
from imutils import face_utils
import time

# Function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1],eye[5])
	B = dist.euclidean(eye[2],eye[4])
	C = dist.euclidean(eye[0],eye[3])

	ear = (A+B)/(2*C)

	return ear

# Function to play the alarm sound
def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)

# Threshold for the eye aspect ratio to be a blink
# The closer the value to 0, the more it shows it's closed
ear_thresh = 0.3

# Number of frames to pass after which it can be considered as the person sleeping
closed_consecutive_frames = 20

# Counter to count the number of frames passed
counter = 0

# Alarm clcok check
alarm_on = False

# Using dlib's hog and linear svm detector to detect faces
detector = dlib.get_frontal_face_detector()

# Using the 68 point facial landmark predictor
predictor = dlib.shape_predictor("/Users/shireen/Documents/opencv/Learning/shape_predictor_68_face_landmarks.dat")

# Getting the indexes of the left eye and right eye out of the 68 points
(lEye_start,lEye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rEye_start,rEye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Starting the camera
camera = cv2.VideoCapture(0)

# Warming up the camera
time.sleep(1)

# Looping through each frame
while True:
	(grabbed,frame) = camera.read()

	if not grabbed:
		break

	frame = imutils.resize(frame,width=400)
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	rects = detector(gray,1)

	# Looping over each detected face
	for rect in rects:
		# Detecting the 68 points on the face
		shape = predictor(gray,rect)
		# Converting the returned points to a numpy array
		shape = face_utils.shape_to_np(shape)

		# Extracting the left and right eye regions
		leftEye = shape[lEye_start:lEye_end]
		rightEye = shape[rEye_start:rEye_end]

		# Calculating the aspect ratios
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# Averaging the aspect ratios
		ear = (leftEAR + rightEAR) / 2

		# Drawing a boundary around the eye
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame,[leftEyeHull],-1,(0,255,0),1)
		cv2.drawContours(frame,[rightEyeHull],-1,(0,255,0),1)


		# Checking if the EAR is less than the threshols
		if ear < ear_thresh:
			counter += 1

			if counter >= closed_consecutive_frames:
				if not alarm_on:
					alarm_on = True
					sound_alarm("/Users/shireen/Documents/opencv/CVProjects/alarm.mp3")

				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		else:
			counter = 0
			alarm_on = False
			cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			

	cv2.imshow("Frame",frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
camera.release()

