
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

dataset = np.loadtxt("dataset.csv", delimiter=',')
x = dataset[:,0:8]
y = dataset[:,8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x, y, epochs=768, batch_size=10)

model_json = model.to_json()
with open("model_test.json", 'w') as json_file:
	json_file.write(model_json)
model.save_weights("model.h5")