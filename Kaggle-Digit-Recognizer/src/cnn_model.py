import numpy
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

class CNNModel(object):
    def __init__(self, model_config=None):
        self.model_config = model_config
        self.cnn = None

    def create(self):
        n_class = self.model_config['n_class']

        self.cnn = Sequential()

        self.cnn.add(Convolution2D(8, 3, 3, border_mode='same', input_shape=(1, 28, 28)))
        self.cnn.add(Activation('relu'))

        self.cnn.add(Convolution2D(16, 3, 3))
        self.cnn.add(Activation('relu'))
        self.cnn.add(MaxPooling2D(pool_size=(2, 2)))
        self.cnn.add(Dropout(0.25))

        self.cnn.add(Convolution2D(32, 3, 3, border_mode='same'))
        self.cnn.add(Activation('relu'))
        self.cnn.add(Convolution2D(64, 3, 3))
        self.cnn.add(Activation('relu'))
        self.cnn.add(MaxPooling2D(pool_size=(2, 2)))
        self.cnn.add(Dropout(0.25))

        self.cnn.add(Flatten())
        self.cnn.add(Dense(512))
        self.cnn.add(Activation('relu'))
        self.cnn.add(Dropout(0.5))
        self.cnn.add(Dense(n_class))
        self.cnn.add(Activation('softmax'))

    def compile(self):
        self.cnn.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        self.cnn.summary()

    def load_trained_model(self):
        self.cnn = keras.models.load_model(self.model_config['trained_model_path'])

    def train(self, x_train, y_train):
        self.create()
        self.compile()
        self.cnn.fit(x_train, y_train,
                     batch_size=self.model_config['batch_size'],
                     nb_epoch=self.model_config['n_epoch'],
                     validation_split=0.01,
                     shuffle=self.model_config['shuffle_train_data'])
        if self.model_config['save_trained_model'] is True:
            self.cnn.save(self.model_config['save_trained_model_path'])

    def predict(self, x_predict):
        y_predict = self.cnn.predict_classes(x_predict)
        return y_predict
