import numpy as np
import pandas as pd
from keras.utils import to_categorical
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense
from sklearn.model_selection import train_test_split
train_set= np.load('/Users/eliecliman/Desktop/dominik/x_train.npy', allow_pickle=True)
test_set = np.load('/Users/eliecliman/Desktop/dominik/x_test2.npy', allow_pickle=True)
train_labels=pd.read_csv('/Users/eliecliman/Desktop/dominik/train_max_y1.csv')

train_labels = np.asarray(train_labels['Label'])
train_labels=to_categorical(train_labels)
print(train_labels.shape)
x_train,x_val,y_train,y_val = train_test_split(train_set,train_labels,test_size=0.15)

def keras_reshape(a):
    return a.reshape(a.shape[0],128,128,1)

x_train = keras_reshape(x_train)
x_val = keras_reshape(x_val)
x_train = x_train/255
x_val = x_val/255

from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Conv2D, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
num_labels = y_train.shape[1]


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
                   optimizer='adam',
                   metrics=['accuracy'] )

    model.save_weights( 'mnistneuralnet.h5' )

    return model


model = create_model()

model.fit(x_train, y_train,
          batch_size=64, epochs=30,
          validation_data=(x_val, y_val))

#prediction = model.predict_categorical(images)[0]
bestclass = ''
bestconf = -1
#for n in [0,1,2,3,4,5,6,7,8,9]:
#       if (prediction[n] > bestconf):
#		bestclass = str(n)
#		bestconf = prediction[n]
#print(bestclass + " " + str(bestconf * 100) + '% confidence.')
