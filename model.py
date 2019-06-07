from keras.models import Model
from keras.datasets import mnist
from keras.layers import Dense, UpSampling2D, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape
import sys
from keras import backend as K
import matplotlib as plt
ACTIVATION = 'relu'
(x_train, _), (x_test, _) = mnist.load_data()


batch_size = 128
epochs = 12

img_rows, img_cols = 28, 28

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(8, (3, 3), activation=ACTIVATION, padding='same')(input_img)
x = Conv2D(8, (3, 3), activation=ACTIVATION, padding='same')(x)

x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(16, (3, 3), activation=ACTIVATION, padding='same')(x)
x = Conv2D(16, (3, 3), activation=ACTIVATION, padding='same')(x)

x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation=ACTIVATION, padding='same')(x)
x = Conv2D(8, (3, 3), activation=ACTIVATION, padding='same')(x)

encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation=ACTIVATION, padding='same')(encoded)
x = Conv2D(8, (3, 3), activation=ACTIVATION, padding='same')(x)

x = UpSampling2D((2, 2))(x)

x = Conv2D(16, (3, 3), activation=ACTIVATION, padding='same')(x)
x = Conv2D(16, (3, 3), activation=ACTIVATION, padding='same')(x)

x = UpSampling2D((2, 2))(x)

x = Conv2D(16, (3, 3), activation=ACTIVATION, padding='same')(x)
x = Conv2D(16, (3, 3), activation=ACTIVATION, padding='same')(x)

x = UpSampling2D((2, 2))(x)

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))

decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()