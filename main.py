'''Title:Emotion Based Music Recommendation System
Shivam Kumar Jha(M210702CA)
Dikshant Bisht(M210670CA)

Abstract:This Model collect data as landmark points from frame of users face image using mediapipe
It's an image processing project using kesar (RNN)
It will take language and singer name from user and then take users emotions using webcam and then it will recommend songs  on Youtube.
'''

import streamlit as st#for creating a web page
from streamlit_webrtc import webrtc_streamer
import av
import cv2#for opening camera
import numpy as np 
import mediapipe as mp #for taking data from camera
from keras.models import load_model
import webbrowser

#loading model and labels
model  = load_model("model.h5")
label = np.load("labels.npy")

#checking users landmarks face and hands
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

#giving header to web page
st.header("Emotion Based Music Recommender")

if "run" not in st.session_state:
	st.session_state["run"] = "true"

try:
	emotion = np.load("emotion.npy")[0]
except:
	emotion=""

if not(emotion):
	st.session_state["run"] = "true"
else:
	st.session_state["run"] = "false"

class EmotionProcessor:
	def recv(self, frame):
		frm = frame.to_ndarray(format="bgr24")

		#flipping camera
		frm = cv2.flip(frm, 1)

		res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

		lst = []
		#loading landmarks
		if res.face_landmarks:
			for i in res.face_landmarks.landmark:
				lst.append(i.x - res.face_landmarks.landmark[1].x)
				lst.append(i.y - res.face_landmarks.landmark[1].y)

			if res.left_hand_landmarks:
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

			lst = np.array(lst).reshape(1,-1)
			#predicting emotion
			pred = label[np.argmax(model.predict(lst))]
			#printing emotion
			print(pred)
			cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)

			np.save("emotion.npy", np.array([pred]))

		#checking users landmarks
		drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
								landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), thickness=-1, circle_radius=1),
								connection_drawing_spec=drawing.DrawingSpec(thickness=1))
		drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
		drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)



		return av.VideoFrame.from_ndarray(frm, format="bgr24")
#getting language and singer name from user
lang = st.text_input("Language")
singer = st.text_input("singer")

if lang and singer and st.session_state["run"] != "false":
	webrtc_streamer(key="key", desired_playing_state=True,
				video_processor_factory=EmotionProcessor)

#Submit button
btn = st.button("Recommend me songs")

if btn:
	if not(emotion):
		st.warning("Please let me capture your emotion first")
		st.session_state["run"] = "true"
	else:
		#opening youtube
		webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}")
		np.save("emotion.npy", np.array([""]))
		st.session_state["run"] = "false"
