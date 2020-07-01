import numpy as np
import keras
from keras.models import Sequential
import numpy
from keras import losses
from keras import optimizers

from keras.layers import Dense, Flatten

from sklearn.model_selection import train_test_split
#acquiring data from external csv file gathered from data glove
myData = np.genfromtxt('trainingData.csv', delimiter=',')

#splitting data for training
X = myData[:, 0:myData.shape[1]-1]

#splitting labels for training
y = myData[:, -1]

#Splitting data in to a training set and a test set to see the accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=0)

num_classes=24

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

#los,metric
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.1),
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, epochs = 20)    
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss: %.4f'% score[0])
print('Test accuracy %.4f'% score[1])   

# X_valid=X[4409,0:X.shape[1]].reshape(1,-1)
# print(X_valid)
# Y_valid=model.predict(X_valid)
# lang_index = numpy.argmax(Y_valid)
# print(lang_index)
# load json and create model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 