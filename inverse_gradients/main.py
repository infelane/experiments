from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Deconv2D, UpSampling2D
from keras import backend as K

from inverse_gradients.flipGradientTF import GradientReversal


class Main(object):
    def __init__(self):

        self.num_classes = 10
        self.foo1()

    def init_model(self):

        l_in = Input(shape=self.input_shape)
        l = Conv2D(32, kernel_size=(3, 3),
                         activation='relu')(l_in)
        l_enc = Conv2D(64, (3, 3), activation='relu')(l)
        l = MaxPooling2D(pool_size=(2, 2))(l)
        l1 = Dropout(0.25)(l)

        l = Flatten()(l1)
        l = Dense(128, activation='relu')(l)
        l = Dropout(0.5)(l)
        l_out = Dense(self.num_classes, activation='softmax', name='classifier')(l)

        flip = GradientReversal(1)
        l2 = flip(l_enc)
        l2 = Deconv2D(64, (3, 3), activation='elu')(l2)
        l2 = Deconv2D(32, (3, 3), activation='elu')(l2)
        l_out_decoder = Conv2D(1, (1, 1), activation='sigmoid', name='decoder')(l2)

        model = Model(inputs=l_in, outputs=[l_out, l_out_decoder])

        model.compile(loss=[keras.losses.categorical_crossentropy, keras.losses.mse],
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        self.model = model

    def foo1(self):

        batch_size = 128

        epochs = 1

        # input image dimensions
        img_rows, img_cols = 28, 28

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            self.input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            self.input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        self.init_model()

        self.model.fit(x_train, [y_train, x_train],
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, [y_test, x_test]))

        import matplotlib.pyplot as plt


        import numpy as np

        def bar(i):
            x0 = x_test[i:i + 1]
            y0 = self.model.predict(x0)
            y01 = y0[0]
            y02 = y0[1][0, ..., 0]

            plt.figure()
            plt.subplot(2, 1, 1)
            plt.imshow(x0[0, ..., 0])
            plt.title(f'{np.argmax(y_test[i])}')
            plt.subplot(2, 1, 2)
            plt.imshow(y02)
            plt.title(f'{np.argmax(y01)}')
            plt.show()

        bar(0)


        # score = self.model.evaluate(x_test, [y_test, x_test], verbose=0)
        # print('Test loss:', score[0])
        # print('Test accuracy:', score[1])

if __name__ == '__main__':
    Main()
