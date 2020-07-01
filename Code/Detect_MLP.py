import numpy as np
import keras
from keras.models import Sequential
import numpy
from keras import losses
from keras import optimizers
from keras.models import model_from_json
from keras.layers import Dense, Flatten

myData = np.genfromtxt('trainingData.csv', delimiter=',')

#splitting data for training
X = myData[:, 0:myData.shape[1]-1]

#splitting labels for training
y = myData[:, -1]

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")

loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.1),
              metrics=['accuracy'])
print("Loaded model from disk")
X_valid=X[4409,0:X.shape[1]].reshape(1,-1)
print(X_valid)
Y_valid=loaded_model.predict(X_valid)
lang_index = numpy.argmax(Y_valid)
print(lang_index)