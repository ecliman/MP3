import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

from sklearn.model_selection import train_test_split


def load_images(filename):
    return np.load(filename).reshape(-1, 128, 128, 1) / 255


def train_dataset(n=-1):
    images = load_images('data/bin_train_max_x.npy')
    if n != -1:
        images = images[:n]
    print("Train images: ", images.shape)

    labels = pd.read_csv('data/train_max_y.csv', index_col='Id', squeeze=True).to_numpy()
    labels = keras.utils.to_categorical(labels, num_classes=10)
    if n != -1:
        labels = labels[:n]
    print("Train labels: ", labels.shape)

    return train_test_split(images, labels, test_size=0.15)


def create_model():
    model = Sequential()

    model.add(ZeroPadding2D((1,1),input_shape=(128, 128, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='r2elu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
              
    return model


x_train, x_valid, y_train, y_valid = train_dataset()
epochs = 30
batch_size = 200

model = create_model()

callbacks = [
    keras.callbacks.TensorBoard(update_freq='batch', histogram_freq=10, write_graph=True, write_images=True)
]

history = model.fit(x_train, y_train, epochs=epochs,
                    batch_size=batch_size, 
                    verbose=1,
                    validation_data=(x_valid, y_valid),
                    callbacks=callbacks)


x_test = load_images('data/bin_test_max_x.npy')
test_pred = model.predict_classes(x_test, batch_size=batch_size, verbose=1)
pd.DataFrame(test_pred).to_csv("pred.csv", header=["Label"], index_label="Id")

