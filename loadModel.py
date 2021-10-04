
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

dataset = np.loadtxt("dataset.csv", delimiter=',')
x = dataset[:,0:8]
y = dataset[:,8]

json_file = open("model_test.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

_, accuracy = model.evaluate(x, y)
print("Acc: %.2f"%(accuracy*100))

predictions = model.predict(x)
for i in range(5):
		print('%s => %d (expected %d)' % (x[i].tolist(), predictions[i], y[i]))
