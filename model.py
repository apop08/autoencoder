from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, UpSampling2D
from keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape
import sys
from keras import backend as K

ACTIVATION = 'relu'



batch_size = 128
epochs = 12

img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


model = Sequential()
model.add(Conv2D(8, (3, 3), activation=ACTIVATION,
                 padding='same', input_shape=input_shape))
model.add(Conv2D(8, (3, 3), activation=ACTIVATION,
                 padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(16, (3, 3), activation=ACTIVATION,
                 padding='same'))
model.add(Conv2D(16, (3, 3), activation=ACTIVATION,
                 padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(256))
model.add(Dense(784))
model.add(Reshape((16, 7, 7)))
model.add(UpSampling2D(2, 2))
model.add(Conv2D(16, (3, 3), activation=ACTIVATION,
                 padding='same'))
model.add(Conv2D(16, (3, 3), activation=ACTIVATION,
                 padding='same'))
model.add(UpSampling2D(2, 2))
model.add(Conv2D(8, (3, 3), activation=ACTIVATION,
                 padding='same'))
model.add(Conv2D(1, (3, 3), activation=ACTIVATION, padding='same'))
model.compile(optimizer='rmsprop',
              metrics='mse')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])