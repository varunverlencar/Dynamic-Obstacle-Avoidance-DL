""" 
    Author:     Varun Verlencar, Worcester Polytechnic Institute
    email:      vvverlencar@wpi.edu
    Project:    Learning based Dynamic obstacle avoidance for Mobile Robots
    File:       Model_small_1
"""
import numpy
import itertools
from keras.callbacks import ModelCheckpoint,CSVLogger
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
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.cbook
import warnings

warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
import pandas as pd
import numpy as np
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

def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: http://stackoverflow.com/a/25074150/395857 
    By HYRY
    '''
    from itertools import izip
    pc.update_scalarmappable()
    ax = pc.get_axes()
    for p, color, value in izip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: http://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    '''
    Inspired by:
    - http://stackoverflow.com/a/16124677/395857 
    - http://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()   
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell 
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    # resize 
    fig = plt.gcf()
    #fig.set_size_inches(cm2inch(40, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype(np.double) / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_classification_report(classification_report, title='Classification Report ', cmap='RdBu'):
    '''
    Plot scikit-learn classification report.
    Extension based on http://stackoverflow.com/a/31689645/395857 
    '''
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : (len(lines) - 2)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        print(v)
        plotMat.append(v)

    print('plotMat: {0}'.format(plotMat))
    print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)

def classifaction_report_csv(report,f):
    report_data = []
    lines = report.split('\n')
    with open(f + '.csv','wb') as fp:
        for line in lines:
            fp.write(line +',')
            fp.write('\n')


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

shift = 0.12
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
    width_shift_range=shift,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=shift,  # randomly shift images vertically (fraction of total height)
    # horizontal_flip=True,  # randomly flip images
    # vertical_flip=True,
    channel_shift_range = True,
    fill_mode='nearest'
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
    width_shift_range=shift,  # randomly shift images horizontally (fracti
    height_shift_range=shift,  # randomly shift images vertically (fractio
    # horizontal_flip=True,  # randomly flip images
    # vertical_flip=True,
    channel_shift_range = True,
    fill_mode='nearest'
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
    '../../Dataset/train',  # this is the target directory
    target_size=(160, 90),
    batch_size=10,
    shuffle = True,
    class_mode='categorical')  
class_dictionary = train_generator.class_indices

print "training data read, classes: "
print class_dictionary


# this is a similar generator, for validation data
validation_generator = validation_datagen.flow_from_directory(
    '../../Dataset/validation',
    target_size=(160, 90),
    batch_size=10,
    shuffle=True,
    class_mode='categorical')

print "validation data read"

# this is a similar generator, for validation data
test_generator = test_datagen.flow_from_directory(  
    '../../Dataset/test',
    target_size=(160, 90),
    batch_size=10,
    shuffle = False,
    class_mode='categorical')

class_dictionary = test_generator.class_indices
class_nb_images = test_generator.nb_sample 

print "test data read, classes: "


samples_per_epoch=2000 #samples_per_epoch
nb_val_samples=800 #nb_val_samples
nb_test_samples = 2855

y_test = np.ones((nb_test_samples),dtype = int)
y_test[0:2220] = 0      #forward
y_test[2220:2570] = 1   #left
y_test[2570:2790] = 2   #right
y_test[2790:2855] = 3   #stop

Y_test = np_utils.to_categorical(y_test,4)

learn_r= 0.00025
dec = 0.0000001
reg = 0.000001
nb_epoch = 50
file_prefix = '1-drop0.3_b10_Aug_lr0.00025_dec0.0000001_reg0.000001_-4HL-64-128-2FC-1024'

# define a simple CNN model
def baseline_model():
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

    # opt = Adam(lr=learn_r, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=dec)
    opt = Nadam(lr=learn_r, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=dec)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# build the model
model = baseline_model()
print "model built"
print model.summary()

csv_logger = CSVLogger('Results/Logs/'+ file_prefix + '_training.csv',separator = ',')

folder  = "Weights/"
ensure_dir(folder)
filepath= folder + file_prefix +  "_weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint,csv_logger]


print 'fitting model'
history = model.fit_generator(
    train_generator,
    samples_per_epoch=samples_per_epoch,
    nb_epoch=nb_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_val_samples,
    verbose = 2,
    callbacks = callbacks_list
    )

vscores = model.evaluate_generator(validation_generator,val_samples = nb_val_samples)
print("Validation Error: %.2f%%, for nb_val_samples=%d samples_per_epoch=%d" % (100-vscores[1]*100,nb_val_samples,samples_per_epoch))

tscores = model.evaluate_generator(test_generator,   val_samples = nb_val_samples)
print("Test Error: %.2f%%, for nb_val_samples=%d samples_per_epoch=%d" % (100-tscores[1]*100,nb_val_samples,samples_per_epoch))


folder  = "Results/Plots/"
ensure_dir(folder)

target_names = ['Forward','Left','Right','Stop']

# summarize history for accuracy
fig = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.grid()
fileName = file_prefix + "_accuracy_val.png"
plt.savefig(folder + fileName, bbox_inches='tight')
plt.close(fig)

# summarize history for loss
fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.grid()
plt.legend(['train', 'val'], loc='lower right')
fileName = file_prefix +  "_loss.png"
plt.savefig(folder + fileName, bbox_inches='tight')
plt.close(fig)

folder  = "Results/Files/"
ensure_dir(folder)
with open(folder + file_prefix + "_Output.txt", "wb") as text_file:
    text_file.write("Using Opt=Nadam,batch = 10, (160x90) lr =  %.8f, decay =  %.8f, reg =  %.8f\n  Validation Error: %.2f%%, Test Error: %.2f%%, for nb_val_samples=%d samples_per_epoch=%d" %(learn_r,dec,reg,100-vscores[1]*100,100-tscores[1]*100,nb_val_samples,samples_per_epoch))

out_pred = model.predict_generator(test_generator,nb_test_samples)
print '\nY_test:',Y_test
print '\nOutPredict:',out_pred

target_names = ['Forward','Left','Right','Stop']

# Plot normalized confusion matrix
folder  = "Results/Plots/ConfusionMatrix/"
cm = confusion_matrix(np.argmax(Y_test,axis=1), np.argmax(out_pred,axis=1))

cnf_matrix = plt.figure()
plot_confusion_matrix(cm, classes=target_names, normalize=True, title='Normalized confusion matrix')
cnf_matrix.savefig(folder + file_prefix +'_ConfusionMatrix' + '.png', dpi=300, format='png', bbox_inches='tight')
plt.close(cnf_matrix)


#Plot classification report
folder = 'Results/Plots/ClassificationReport/'
report = classification_report(np.argmax(Y_test,axis=1), np.argmax(out_pred,axis=1),target_names=target_names)

plot_classification_report(report)
plt.savefig(folder + file_prefix +'_ClassificationReport.png', dpi=300, format='png', bbox_inches='tight')
plt.close()

folder ='Results/Files/Classification_report/'
ensure_dir(folder)

print '\n*Classification Report:\n', report
classifaction_report_csv(report,folder + file_prefix)

