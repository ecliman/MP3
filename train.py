import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Conv2D, Dropout
from keras.layers import Flatten, Dense
from keras import optimizers

from sklearn.model_selection import train_test_split

# ** Model Begins **

def create_model():
    model = Sequential()
    model.add(ZeroPadding2D( (1, 1), input_shape=(128, 128, 1) ) )
    model.add((Conv2D( 32, kernel_size=(3, 3), activation='relu' )) )
    model.add((ZeroPadding2D((1, 1))))
    model.add((Conv2D( 32, kernel_size=(3, 3), activation='relu' )) )
    model.add(MaxPooling2D( 2, 2 ) )

    model.add(ZeroPadding2D( (1, 1), input_shape=(128, 128, 1) ) )
    model.add((Conv2D( 64, kernel_size=(3, 3), activation='relu' )) )
    model.add(ZeroPadding2D( (1, 1) ) )
    model.add((Conv2D( 64, kernel_size=(3, 3), activation='relu' )) )
    model.add((MaxPooling2D(2, 2)))
    model.add( Dropout( 0.5 ) )

    model.add( ZeroPadding2D( (1, 1), input_shape=(128, 128, 1) ) )
    model.add( (Conv2D( 128, kernel_size=(3, 3), activation='relu' )) )
    model.add( ZeroPadding2D( (1, 1) ) )
    model.add( (Conv2D( 128, kernel_size=(3, 3), activation='relu' )) )
    model.add( MaxPooling2D( 2, 2 ) )
    model.add( Dropout( 0.5 ) )

    model.add( ZeroPadding2D( (1, 1), input_shape=(128, 128, 1) ) )
    model.add( (Conv2D( 256, kernel_size=(3, 3), activation='relu' )) )
    model.add( ZeroPadding2D( (1, 1) ) )
    model.add( (Conv2D( 256, kernel_size=(3, 3), activation='relu' )) )
    model.add((MaxPooling2D(2, 2)))
    model.add( Dropout( 0.5 ) )

    model.add( Flatten() )
    model.add( Dense( 512, activation='relu' ) )
    model.add(Dropout(0.5))
    model.add( Dense( 128, activation='relu' ) )
    model.add( Dense( 10, activation='softmax' ) )
    # ** Model Ends **

    model.summary()

    model.compile( loss='categorical_crossentropy',
                   optimizer=optimizers.Adam(learning_rate=0.01),
                   metrics=['accuracy'] )

    return model

labels = pd.read_csv('data/train_max_y.csv', index_col='Id', squeeze=True).to_numpy() #nrows=64*100
labels = keras.utils.to_categorical(labels, num_classes=10)
labels = labels[:12800]

#x_dt = np.dtype( (np.uint8, (128, 128, 1)) )
#images = np.fromfile( 'data/raw_train_max_x', dtype=x_dt, count=batch_size*100)

def read_images(filename):
    images = pd.read_pickle(filename).reshape((-1, 128, 128, 1))
#    images[images < 255.0] = 0.0
#    images = images.astype(np.uint8)
    images = images / 255
    return images

images = read_images('data/train_max_x')
images = images[:12800]

x_train, x_valid, y_train, y_valid = train_test_split(images, labels, test_size=0.15, shuffle=True)

model = create_model()
model.fit(x_train, y_train,
          batch_size=64, epochs=30,
          validation_data=(x_valid, y_valid))


model.save_weights('tf_model_weights.h5')

#test_images = np.fromfile('data/raw_test_max_x', dtype=x_dt)
test_images = read_images('data/test_max_x')

test_pred = model.predict(test_images)
print(test_pred[0])
pd.DataFrame(test_pred.argmax(axis=1)).to_csv("pred.csv", header=["Label"], index_label="Id")


