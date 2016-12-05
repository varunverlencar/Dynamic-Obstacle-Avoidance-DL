#Varun Verlencar
#vvverlencar@wpi.eduu
#WPI-Hand Gestures
import numpy
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.regularizers import l2, activity_l2
from keras.optimizers import Nadam
from keras.layers.noise import GaussianNoise
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils.visualize_util import plot

import csv
import os
import h5py
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as K
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

def ensure_dir(f):
	d = os.path.dirname(f)
	if not os.path.exists(d):
		os.makedirs(d)

shift = 0.1
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
	rescale=1./255,
	# featurewise_center= True,  # set input mean to 0 over the dataset
	# samplewise_center=True,  # set each sample mean to 0
	# featurewise_std_normalization=True,  # divide inputs by std of the dataset
	# samplewise_std_normalization=True,  # divide each input by its std
	# zca_whitening=True,  # apply ZCA whitening
	# zoom_range=0.3,
	# rotation_range=90,  # randomly rotate images in the range (degrees, 0 to 180)
	# width_shift_range=shift,  # randomly shift images horizontally (fraction of total width)
	# height_shift_range=shift,  # randomly shift images vertically (fraction of total height)
	# horizontal_flip=True,  # randomly flip images
	# vertical_flip=True,
	# fill_mode='nearest'
	)

# this is the augmentation configuration usde for validation
validation_datagen = ImageDataGenerator(
	rescale=1./255,
	# featurewise_center= True,  # set input mean to 0 over the dataset
	# samplewise_center=True,  # set each sample mean to 0
	# featurewise_std_normalization=True,  # divide inputs by std of the dataset
	# samplewise_std_normalization=True,  # divide each input by its std
	# zca_whitening=True,  # apply ZCA whitening
	# zoom_range=0.3,
	# rotation_range=90,  # randomly rotate images in the range (degrees, 0
	# width_shift_range=shift,  # randomly shift images horizontally (fracti
	# height_shift_range=shift,  # randomly shift images vertically (fractio
	# horizontal_flip=True,  # randomly flip images
	# vertical_flip=True,
	# fill_mode='nearest'
	)
	

# this is the augmentation configuration used for testing
test_datagen = ImageDataGenerator(
	rescale=1./255
	
	# zca_whitening=True,  # apply ZCA whitening
	)

# this is a generator that will read pictures found in
# subfolers of 'dataset/train', and indefinitely generate
# batches of augmented image data

train_generator = train_datagen.flow_from_directory(
	'../Dataset/train',  # this is the target directory
	target_size=(320, 180),
	batch_size=50,
	shuffle = True,
	class_mode='categorical')  

print "training data read"

# this is a similar generator, for validation data
validation_generator = validation_datagen.flow_from_directory(
	'../Dataset/validation',
	target_size=(320, 180),
	batch_size=50,
	shuffle = True,
	class_mode='categorical')

print "validation data read"

# this is a similar generator, for validation data
test_generator = test_datagen.flow_from_directory(
	'../Dataset/test',
	target_size=(320, 180),
	batch_size=50,
	shuffle = True,
	class_mode='categorical')

print "test data read"

learn_r= 0.0001
dec = 0.0000005
reg = 0.000001
#     learn_r= 0.0005
# dec = 0.00000000001
# reg = 0.00000000001

# define a simple CNN model
def baseline_model():
	# create model
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(3,320,180)))     #0
	first_layer = model.layers[-1]
	# this is a placeholder tensor that will contain our generated images
	input_img = first_layer.input

	model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1', W_regularizer = l2(reg)))
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
	model.add(Dense(128	, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(128	, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(4, activation='softmax'))

	# opt = Adam(lr=learn_r, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=dec)
	opt = Nadam(lr=learn_r, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=dec)
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# build the model
model = baseline_model()
print "model built"
print model.summary()

folder  = "Images/Model/Aug/"
ensure_dir(folder)
plot(model, to_file='2-4HL-2FC_Aug_model.png')

i=1000 #samples_per_epoch
j=800 #nb_val_samples

folder  = "Weights/Best/Aug/"
ensure_dir(folder)
filepath= folder + "2-4HL-2FC_Aug_weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


print 'fitting model'
history = model.fit_generator(
	train_generator,
	samples_per_epoch=i,
	nb_epoch=15,
	validation_data=validation_generator,
	nb_val_samples=j,
	verbose = 2,
	callbacks = callbacks_list
	)

# folder  = "Aug/Weights/main/"
# ensure_dir(folder)
# model.save_weights( folder +'first_try.h5')

vscores = model.evaluate_generator(validation_generator,val_samples = j)
print("Validation Error: %.2f%%, for nb_val_samples=%d samples_per_epoch=%d" % (100-vscores[1]*100,j,i))

tscores = model.evaluate_generator(test_generator,
	val_samples = j)
print("Test Error: %.2f%%, for nb_val_samples=%d samples_per_epoch=%d" % (100-tscores[1]*100,j,i))


folder  = "Images/Aug/"
ensure_dir(folder)

# summarize history for accuracy
fig = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
fileName = "2-4HL-2FC_Aug_accuracy_val.png"
plt.savefig(folder + fileName, bbox_inches='tight')
plt.close(fig)

# summarize history for loss
fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
fileName = "2-4HL-2FC_Aug_loss.png"
plt.savefig(folder + fileName, bbox_inches='tight')
plt.close(fig)

folder  = "Files/"
ensure_dir(folder)
with open(folder +"2-Output_Aug_4HL-2FC.txt", "wb") as text_file:
	text_file.write("Using Opt=Nadam, lr =  %.8f, decay =  %.8f, reg =  %.2f\n  Validation Error: %.2f%%, Test Error: %.2f%%, for nb_val_samples=%d samples_per_epoch=%d" %(learn_r,dec,reg,100-vscores[1]*100,100-tscores[1]*100,j,i))


