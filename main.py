
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

model.fit(x, y, epochs=150, batch_size=10)

_, accuracy = model.evaluate(x, y)
print("Acc: %.2f"%(accuracy*100))

predictions = model.predict(x)
for i in range(5):
		print('%s => %d (expected %d)' % (x[i].tolist(), predictions[i], y[i]))