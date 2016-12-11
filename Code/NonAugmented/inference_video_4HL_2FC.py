from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD
from keras import backend as K
K.set_image_dim_ordering('th')
import cv2, numpy as np
import pickle
   
def output_label(out): 
    if out == 0:
        return 'Forward'
    elif out == 1:
        return 'Left'
    elif out == 2:
        return 'Right'
    elif out == 3:
        return 'Stop'
    elif:
        return 'None'
    
def baseline_model(weights_path=None):
    # create model
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))     #0
    first_layer = model.layers[-1]
    # this is a placeholder tensor that will contain our generated images
    input_img = first_layer.input

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1',W_regularizer = l2(reg)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(128 , activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128 , activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(5, activation='softmax'))

    
    return model

    if weights_path:
        model.load_weights(weights_path)

    return model

if __name__ == "__main__":

    learn_r= 0.0001
    dec = 0.0000005
    reg = 0.0001

    # Test pretrained model
    model = baseline_model('Weights/Best/main/1-4HL-2FC_weights.best.hdf5')
    opt = Adam(lr=learn_r, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=dec)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    
    ########################
    # REAL-TIME PREDICTION #
    ########################

    print '... Initializing RGB stream'
    
     #### Initialize built-in webcam
    cap = cv2.VideoCapture('../../../Dataset Videos_classified/Forward/Forward_bg1_obj1-1.mp4')
    # Enforce size of frames
    cap.set(3, 320) 
    cap.set(4, 180)

    shot_id = 0
 
    #### Start video stream and online prediction
    while (True):
         # Capture frame-by-frame
    
#        start_time = time.clock()
        
        ret, frame = cap.read()
        
        #color_frame = color_stream.read_frame() ## VideoFrame object
        #color_frame_data = frame.get_buffer_as_uint8() ## Image buffer
        #frame = convert_frame(color_frame_data, np.uint8) ## Generate BGR frame
                
        im = cv2.resize(frame, (320, 180)).astype(np.float32)
        # im[:,:,0] -= 103.939
        # im[:,:,1] -= 116.779
        # im[:,:,2] -= 123.68
        im = im.transpose((2,0,1))
        im = np.expand_dims(im, axis=0)
        
        out = model.predict(im)
        # output_label(out)
        #print np.argmax(out)
        #print my_list[np.argmax(out)]
        
        # we need to keep in mind aspect ratio so the image does
        # not look skewed or distorted -- therefore, we calculate
        # the ratio of the new image to the old image
        #r = 100.0 / frame.shape[1]
        dim = (640, 360)
 
        # perform the actual resizing of the image and show it
        resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        
        # Write some Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(resized,output_label(np.argmax(out)),(20,450), font, 1, (255,255,255),1,1)
        # Display the resulting frame
        cv2.imshow('Indoor Obstacle Avoidance',resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()