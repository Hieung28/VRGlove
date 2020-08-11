import numpy as np
import keras
from keras.models import Sequential
import numpy
from keras import losses
from keras import optimizers
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten

from sklearn.model_selection import train_test_split
#acquiring data from external csv file gathered from data glove
myData = np.genfromtxt('CombinedHandData.csv', delimiter=',')

#splitting data for training
X = myData[:, 0:myData.shape[1]-1]

#splitting labels for training
y = myData[:, -1]

#Splitting data in to a training set and a test set to see the accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=0)

num_classes=27

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train)
model = Sequential()

#model.add(Dense(128, activation='relu'))
model.add(Dense(256,input_shape=(44,), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

#los,metric
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])

history=model.fit(X_train, y_train, batch_size=512, epochs = 40)    
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss: %.4f'% score[0])
print('Test accuracy %.4f'% score[1])   

X_valid=X[28001,0:X.shape[1]].reshape(1,-1)
print(X_valid)
Y_valid=model.predict(X_valid)
print(Y_valid)
lang_index = numpy.argmax(Y_valid)
print(lang_index)
# load json and create model
model_json = model.to_json()
with open("twohandModel.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("twohandModel.h5")
print("Saved model to disk")

# summarize history for accuracy
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()