import os  
import numpy as np 
import cv2 
from tensorflow.keras.utils import to_categorical
#importing libraries for our model
from keras.layers import Input, Dense 
from keras.models import Model
 
is_init = False
size = -1

label = []
#every emotions will be convert to unique integer
dictionary = {}
c = 0
#iterating all files in directory
for i in os.listdir():
	#opening all emotions files
	if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):  
		if not(is_init):
			is_init = True
			#loading file into X
			X = np.load(i)
			#getting rows
			size = X.shape[0]
			y = np.array([i.split('.')[0]]*size).reshape(-1,1)
		else:
			X = np.concatenate((X, np.load(i)))
			y = np.concatenate((y, np.array([i.split('.')[0]]*size).reshape(-1,1)))

		label.append(i.split('.')[0])
		dictionary[i.split('.')[0]] = c  
		c = c+1


for i in range(y.shape[0]):
	y[i, 0] = dictionary[y[i, 0]]
#changing data type to integer
y = np.array(y, dtype="int32")

#conerting y to categorical data
y = to_categorical(y)

#shuffling data
X_new = X.copy()
y_new = y.copy()
counter = 0 

cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)
#new shuffled data
for i in cnt: 
	X_new[counter] = X[i]
	y_new[counter] = y[i]
	counter = counter + 1

#converting our data to input for our model
ip = Input(shape=(X.shape[1]))

m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)

op = Dense(y.shape[1], activation="softmax")(m)
#running RNN model on   out data
model = Model(inputs=ip, outputs=op)

model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])
#fittinf out model
model.fit(X, y, epochs=50)

#saving our model
model.save("model.h5")
np.save("../labels.npy", np.array(label))
