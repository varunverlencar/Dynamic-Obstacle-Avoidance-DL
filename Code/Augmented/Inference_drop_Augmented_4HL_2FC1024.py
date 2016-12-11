from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import Adam
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as K
K.set_image_dim_ordering('th')
K.set_learning_phase(0)
from keras.regularizers import l2
from keras.optimizers import Nadam
import cv2, numpy as np
from sklearn.metrics import confusion_matrix,classification_report

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

def plot_weights(layer, x, y):
    """
    Plot wieght vectors after specified conv layer number,
    e.g if we have 20 weigth vectors for layer 0, call plot_weight(l_conv1,5,4)
    to get a 5 x 4 grid plot
    """
    w_vectors  = layer.W.get_value()
    fig =  plt.figure()
    for i in range(len(w_vectors)):
        img = fig.add_subplot(y, x, i+1)
        img.matshow(w_vectors[i][0])      #for Conv1-filter
        # img.matshow(filters[i][1])    #for Conv2-filter
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.tight_layout()
    return plt

def plot_layer_output(layer,img,x,y):
    output_layer = layer.output
    output_fn = K.function([model.layers[0].input],[output_layer])
    
    print '2:', (img.shape)

    output_img = output_fn([img])
    print '3:',(output_img[0].shape)
    
    output_img = np.rollaxis(np.rollaxis(output_img[0],2,1),3,1)
    print '4:',(output_img.shape)
    
    fig = plt.figure(figsize = (128,72))
    for i in range(len(layer.W.get_value())):
        ax = fig.add_subplot(x, y, i+1)
        img = output_img[0,:,:,i]

        ax.imshow(img,cmap='gray', interpolation='nearest')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
    plt.show()
    return fig

def baseline_model(weights_path=None):
    # create model
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,160,90)))     #0
    
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
    Y_test = ['Forward','Left','Right','Stop'] 

    testpath = '../testImages/'
    filenames = [f for f in listdir(testpath) if isfile(join(testpath, f))]
    filenames = ['Stop_bg2_obj300044.jpg']

    for file in filenames:
        print 'File name',file

        f = open('Results/Inference/OutputFiles/' +  file.split('.')[0] +  "-Inference_Images_2-Output_Aug_4HL-2FC1024.txt",'wb')
        im = cv2.imread(testpath + file)
        
        b,g,r = cv2.split(im)       # get b,g,r
        im = cv2.merge([r,g,b])     # switch it to rgb
        
        # plt.imshow(im)
        # plt.show()        
        
        im = cv2.resize(im, (160, 90)).astype(np.float32)
        # im[:,:,0] -= 103.939
        # im[:,:,1] -= 116.779
        # im[:,:,2] -= 123.68
        # plt.imshow(im)
        # plt.show()

        im = im.transpose((2,1,0))
        
        # plt.imshow(im)
        # plt.show()

        im = np.expand_dims(im, axis=0)
        print "1:",(im.shape)
        
        learn_r = 0.00025
        dec = 0.0000001
        reg = 0.000001


        # Test pretrained model
        model = baseline_model('../Weights/Best/Aug/2-drop-Aug-4HL-2FC1024_weights.best.hdf5')
        print model.summary()

        # opt = Adam(lr=learn_r, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=dec)
        opt = Nadam(lr=learn_r, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=dec)
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        out = model.predict(im)

        target_names = Y_test
        cm = confusion_matrix([0,0,0,1], out)
        plt.matshow(cm)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

        print out
        print np.argmax(out)
        print output_label(np.argmax(out))
        f.write(file + '    Predicted Output: ' + output_label(np.argmax(out)) + '\n')
        f.write("drop fc =1024 Using Opt=Nadam,batch = 10, (160x90) lr =  %.8f, decay =  %.8f, reg =  %.8f\n  , for nb_val_samples=800 samples_per_epoch=1000" %(learn_r,dec,reg))
        
        #plot output form intermediate layers
        # plt.close(cm)
        # plot_layer_output(model.layers[1],im,8,8).savefig('Results/Inference/Visualisations/' + file.split('.')[0] + '_Conv_1.jpg', bbox_inches='tight')
        # plt.close()
        # plot_layer_output(model.layers[3],im,8,8).savefig('Results/Inference/Visualisations/' + file.split('.')[0] + '_Conv_2.jpg', bbox_inches='tight')
        # plt.close()
        # plot_layer_output(model.layers[6],im,16,8).savefig('Results/Inference/Visualisations/' + file.split('.')[0] + '_Conv_3.jpg', bbox_inches='tight')
        # plt.close()
        # plot_layer_output(model.layers[8],im,16,8).savefig('Results/Inference/Visualisations/' + file.split('.')[0] + '_Conv_4.jpg', bbox_inches='tight')
        # plt.close()
        
        # plot_weights
    f.close
