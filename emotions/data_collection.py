import mediapipe as mp# libraray for predicting all the land marks of face & hand at very
import numpy as np
import cv2#for VideoCapture

#using cv2 to open camera
cap = cv2.VideoCapture(0)

#enter name of emotion
name = input("Enter the name of the data : ")

#holistic class have the sol to take in frame and return all the land mark of full video
#it will take frame and will return all facial and hands keypoints
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# list for collection of rows
X = []
data_size = 0

while True:
	# list for collection of landmarks(hand,face) in a row and having 1020 col for all landmarks
	lst = []
	# video capture class
	_, frm = cap.read()
#flip left to right for avoiding mirror effect
	frm = cv2.flip(frm, 1)

# conversion frame in RGB from cv2-BGR #process will take frame and return RGB
	res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

#if result is not null
	if res.face_landmarks:
		for i in res.face_landmarks.landmark:
			# Above we are storing all x,y point wrt. any particular land marks
			# point instead of origin to avoid duplicate sample when face/hand is moving with same exp
			#subtracting for different positions
			lst.append(i.x - res.face_landmarks.landmark[1].x)
			lst.append(i.y - res.face_landmarks.landmark[1].y)

		if res.left_hand_landmarks: #if left hand is not in frame just store 0.0
			for i in res.left_hand_landmarks.landmark:
				lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
				lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
		else:
			for i in range(42):
				lst.append(0.0)

		if res.right_hand_landmarks:
			for i in res.right_hand_landmarks.landmark:
				lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
				lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
		else:
			for i in range(42):
				lst.append(0.0)


		X.append(lst)
		data_size = data_size+1


#drawing keypoints on landmarks
	drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
	drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
	drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

	#how many datasize we have collected
	cv2.putText(frm, str(data_size), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

	cv2.imshow("window", frm)

	if cv2.waitKey(1) == 27 or data_size>99:
		cv2.destroyAllWindows()
		cap.release()
		break

#saving emotion file as name
np.save(f"{name}.npy", np.array(X))