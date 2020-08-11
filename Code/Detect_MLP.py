import numpy as np
import keras
from keras.models import Sequential
import numpy
from keras import losses
from keras import optimizers
from keras.models import model_from_json
from keras.layers import Dense, Flatten

myData = np.genfromtxt('trainingData.csv', delimiter=',')
jointAngle=[-0.161, 0.755, -3.11, 4.638, -3.256, -0.195, 0.644, 0.543, 0.292, 0.292, 0.3, 0.495, 0.144, 0.144, -0.11, 0.378, -0.014, -0.014, -0.462, 0.198, 0.383, 0.383]
#splitting data for training
X = myData[:, 0:myData.shape[1]-1]

#splitting labels for training
y = myData[:, -1]

json_file = open('newModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("newModel.h5")

loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])
print("Loaded model from disk")
X_valid=np.array(jointAngle).reshape(1, -1)
print(X_valid)
Y_valid=loaded_model.predict(X_valid)
lang_index = numpy.argmax(Y_valid)
print(lang_index)