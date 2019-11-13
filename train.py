
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Conv2D, Dropout
from keras.layers import Flatten, Dense
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

##########   LOAD FILES ###############
labels = pd.read_csv( '/Users/eliecliman/Desktop/dominik/train_max_y.csv', index_col='Id', squeeze=True,
                      nrows=1000 ).to_numpy()
labels = keras.utils.to_categorical( labels, num_classes=10 )

x_dt = np.dtype( (np.uint8, (128, 128, 1)) )

images = np.fromfile( '/Users/eliecliman/Desktop/dominik/raw_train_max_x', dtype=x_dt, count=1000 )
print("images", images)

x_train, x_valid, y_train, y_valid = train_test_split( images, labels, test_size=0.2, shuffle=True )

############## CREATE MODEL AND HYPER-PARAMETERS ###############

def create_model():
    model = Sequential()
    model.add( ZeroPadding2D( (1, 1), input_shape=(128, 128, 1) ) )
    model.add( (Conv2D( 32, kernel_size=(3, 3), activation='relu' )) )
    model.add( (ZeroPadding2D( (1, 1) )) )
    model.add( (Conv2D( 32, kernel_size=(3, 3), activation='relu' )) )
    model.add( MaxPooling2D( 2, 2 ) )

    model.add( ZeroPadding2D( (1, 1) ) )
    model.add( (Conv2D( 64, kernel_size=(3, 3), activation='relu' )) )
    model.add( ZeroPadding2D( (1, 1) ) )
    model.add( (Conv2D( 64, kernel_size=(3, 3), activation='relu' )) )
    model.add( (MaxPooling2D( 2, 2 )) )
    model.add( Dropout( 0.5 ) )

    model.add( ZeroPadding2D( (1, 1) ) )
    model.add( (Conv2D( 128, kernel_size=(3, 3), activation='relu' )) )
    model.add( ZeroPadding2D( (1, 1) ) )
    model.add( (Conv2D( 128, kernel_size=(3, 3), activation='relu' )) )
    model.add( MaxPooling2D( 2, 2 ) )
    model.add( Dropout( 0.5 ) )

    model.add( ZeroPadding2D( (1, 1) ) )
    model.add( (Conv2D( 256, kernel_size=(3, 3), activation='relu' )) )
    model.add( ZeroPadding2D( (1, 1) ) )
    model.add( (Conv2D( 256, kernel_size=(3, 3), activation='relu' )) )
    model.add( MaxPooling2D( 2, 2 ) )
    model.add( Dropout( 0.5 ) )

    model.add(Flatten())
    model.add( Dense( 512, activation='relu' ) )
    model.add( Dropout( 0.5 ) )
    model.add( Dense(128, activation='relu' ) )
    model.add( Dense( 10, activation='softmax' ) )
    # ** Model Ends **

    model.summary()

    ####FLATTEN###
    model.compile( loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'] )

    model.save_weights( 'mnistneuralnet.h5' )

    return model

############## DRIVER ##################
model = create_model()

model.fit( x_train, y_train,
           batch_size=64, epochs=31,
           validation_data=(x_valid, y_valid) )

prediction = model.predict( images )[0]
bestclass = ''
bestconf = -1
for n in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    if (prediction[n] > bestconf):
        bestclass = str( n )
        bestconf = prediction[n]
print( bestclass + " " + str( bestconf * 100 ) + '% confidence.' )
