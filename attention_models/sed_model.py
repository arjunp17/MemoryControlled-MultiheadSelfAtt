randomSeed=1 # seed value
print('seed value', randomSeed)
import os
os.environ['PYTHONHASHSEED']=str(randomSeed) # Set `PYTHONHASHSEED` environment variable at a fixed value
import numpy as np
np.random.seed(randomSeed) # NumPy
import random
random.seed(randomSeed)    # Python
import tensorflow as tf
tf.set_random_seed(randomSeed) # Tensorflow

from keras.backend.tensorflow_backend import set_session # Configure a new global `tensorflow` session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement =False # True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)   # set this TensorFlow session as the default session for Keras

#############################################################################################################

from keras.models import Model
from keras.layers import Input, merge, Multiply
from keras.layers.core import Flatten, Dense, Dropout, Reshape, Activation, Permute, Lambda, RepeatVector
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling1D, Conv1D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
import keras
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional
from itertools import repeat
from keras_self_attention import SeqSelfAttention ## To install SeqSelfAttention refer to - https://github.com/CyberZHG/keras-self-attention
from keras.initializers import glorot_uniform



#initializer
def init_weights():
	return glorot_uniform(seed=randomSeed)



## baseline model
def baselineSED(freq,time,channels,nb_classes,lr,attention_width,history_only):
	input_shape=(freq,time,channels)
	input = Input(shape=input_shape)	
	# conv1
	conv2d_1 = Conv2D(64, (5, 5), strides=(1, 1), activation='relu', padding='same', kernel_initializer=init_weights())(input)
	conv2d_1 = BatchNormalization(axis=-1)(conv2d_1)
	conv2d_1 = Dropout(0.30, seed=randomSeed)(conv2d_1)
	#maxpool1
	MP_1 = MaxPooling2D(pool_size=(5, 1), strides=(5, 1))(conv2d_1)
	#conv2
	conv2d_2 = Conv2D(64, (5, 5), strides=(1, 1), activation='relu', padding='same', kernel_initializer=init_weights())(MP_1)
	conv2d_2 = BatchNormalization(axis=-1)(conv2d_2)
	conv2d_2 = Dropout(0.30, seed=randomSeed)(conv2d_2)
	#maxpool2
	MP_2 = MaxPooling2D(pool_size=(4, 1), strides=(4, 1))(conv2d_2)
	#conv3
	conv2d_3 = Conv2D(64, (5, 5), strides=(1, 1), activation='relu', padding='same', kernel_initializer=init_weights())(MP_2)
	conv2d_3 = BatchNormalization(axis=-1)(conv2d_3)
	conv2d_3 = Dropout(0.30, seed=randomSeed)(conv2d_3)
	#maxpool3
	MP_3 = MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(conv2d_3)
	#reshape
	reshape1 = Reshape((64,-1))(MP_3)
	reshape1 = Permute((2,1))(reshape1)
	#GRU
	gru = GRU(64, activation='tanh', return_sequences=True, kernel_initializer=init_weights())(reshape1)	
	#Memory Controlled Self Attention
	attn = SeqSelfAttention(64, attention_type=SeqSelfAttention.ATTENTION_TYPE_ADD, attention_activation='sigmoid', attention_width=attention_width, history_only=history_only, name = 'self_attn')(gru)
	#aggregation
	sed_out = TimeDistributed(Dense(nb_classes, activation='sigmoid', kernel_initializer=init_weights(), name = 'sed_out'))(attn)
	model = Model(input, sed_out, name='basemodelSED')	
	opt = Adam(lr = lr)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])	
	model.summary()
	return model





# input dimensions  
freq, time , channels = 40,500,1    

#model params
epochs = 200
batch_size =  32
nb_classes = 10
lr = 0.001
#decay = 1e-6
attention_width = 50
history_only = False

baselineSED = baselineSED(freq,time,channels,nb_classes,lr,attention_width,history_only)



# feature and label
X_train = np.load('../train_feature.npy')
Y_train = np.load('../train_label_SED.npy')
X_val = np.load('../val_feature.npy')
Y_val = np.load('../val_label_SED.npy')


#Model path
modelpath = "...../model/" + str(randomSeed)

## training
def model_train(model,X_train,Y_train,X_val,Y_val,epochs,batch_size,modelpath):
	os.makedirs(modelpath)
	tensorboard = TensorBoard(log_dir=modelpath + "/tboard/")
	filepath = modelpath + "/best_weights.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
	csv_logger = keras.callbacks.CSVLogger(modelpath + "/training.log")
	earlyStopper= keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=2, mode='auto')
	callbacks_list = [checkpoint,tensorboard,csv_logger,earlyStopper]
	history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),callbacks=callbacks_list,epochs=epochs, shuffle=False,batch_size=batch_size,verbose=2)
	return history

	
	
model_history = model_train(baselineSED,X_train,Y_train,X_val,Y_val,epochs,batch_size,modelpath)
