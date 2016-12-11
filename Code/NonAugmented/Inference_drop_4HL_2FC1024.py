from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.regularizers import l2, activity_l2
from keras.optimizers import Nadam
import cv2, numpy as np
import pickle
from os import listdir
from os.path import isfile, join

def output_label(out): 
    if out == 0:
        return 'Forward'
    elif out == 1:
        return 'Left'
    elif out == 2:
        return 'Right'
    elif out == 3:
        return 'Stop'
    else:
        return 'None'

def baseline_model(weights_path=None):
    # create model
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,160,90)))     #0
    first_layer = model.layers[-1]
    # this is a placeholder tensor that will contain our generated images
    input_img = first_layer.input

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1', W_regularizer = l2(reg)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(4, activation='softmax'))


    if weights_path:
        model.load_weights(weights_path)

    return model

if __name__ == "__main__":
    #with open(the_filename, 'wb') as f:
    #    pickle.dump(my_list, f)
    # with open(the_filename, 'rb') as f:
    #     my_list = pickle.load(f)
    testpath = 'testImages/'
    filenames = [f for f in listdir(testpath) if isfile(join(testpath, f))]
    
    f = open('testOutput/' + "Output1.txt",'wb')

    for file in filenames:
        im = cv2.resize(cv2.imread(testpath + file), (90, 160)).astype(np.float32)
        # im[:,:,0] -= 103.939
        # im[:,:,1] -= 116.779
        # im[:,:,2] -= 123.68
        # plt.imshow(im)
        # plt.show()

        im = im.transpose((2,0,1))
        im = np.expand_dims(im, axis=0)

        learn_r= 0.00025
        dec = 0.0000001
        reg = 0.000001

        # Test pretrained model
        model = baseline_model('Weights/Best/Aug/2-drop-Aug-4HL-2FC1024_weights.best.hdf5')
        # opt = Adam(lr=learn_r, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=dec)
        opt = Nadam(lr=learn_r, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=dec)
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        out = model.predict(im)

        print out
        print np.argmax(out)
        print output_label(np.argmax(out))
        f.write(file + '    Output: ' + output_label(np.argmax(out)) + '\n')
    f.close
