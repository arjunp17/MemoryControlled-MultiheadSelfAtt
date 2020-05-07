from keras.models import Model
from keras.layers import Input, merge, Multiply
from keras.layers.core import Flatten, Dense, Dropout, Reshape, Activation, Permute, Lambda, RepeatVector
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling1D, Conv1D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
import keras
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional
from itertools import repeat


# input dimensions  
rows, cols , channels = 40,500,1    

#model params
epochs = 200
batch_size =  32
nb_classes = 10
lr = 0.001
decay = 1e-6
initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2)

## baseline model
def baselineSED(rows,cols,channels,nb_classes,lr):
	
	# conv1
	conv2d_1 = Conv2D(64, (5, 5), strides=(1, 1), activation='relu', padding='same', kernel_initializer=initializer)(input = Input(shape=(rows,cols,channels)))
	conv2d_1 = BatchNormalization(axis=-1)(conv2d_1)
	conv2d_1 = Dropout(0.30)(conv2d_1)

	#maxpool1
	MP_1 = MaxPooling2D(pool_size=(5, 1), strides=(5, 1))(conv2d_1)

	#conv2
	conv2d_2 = Conv2D(64, (5, 5), strides=(1, 1), activation='relu', padding='same', kernel_initializer=initializer)(MP_1)
	conv2d_2 = BatchNormalization(axis=-1)(conv2d_2)
	conv2d_2 = Dropout(0.30)(conv2d_2)

	#maxpool2
	MP_2 = MaxPooling2D(pool_size=(4, 1), strides=(4, 1))(conv2d_2)

	#conv3
	conv2d_3 = Conv2D(64, (5, 5), strides=(1, 1), activation='relu', padding='same', kernel_initializer=initializer)(MP_2)
	conv2d_3 = BatchNormalization(axis=-1)(conv2d_3)
	conv2d_3 = Dropout(0.30)(conv2d_3)

	#maxpool3
	MP_3 = MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(conv2d_3)

	#reshape
	reshape1 = Reshape((64,-1))(MP_3)
	reshape1 = Permute((2,1))(reshape1)

	#GRU
	gru = GRU(64, activation='tanh', return_sequences=True)(reshape1)
	
	#Memory Controlled Self Attention
	attn = SeqSelfAttention(64, attention_type=SeqSelfAttention.ATTENTION_TYPE_ADD, attention_activation='sigmoid', attention_width=50, history_only=False, name = 'self_attn')(gru)

	#aggregation
	sed_out = TimeDistributed(Dense(nb_classes, activation='sigmoid', name = 'sed_out'))(attn)

	model = Model(input, sed_out, name='basemodelSED')	
	opt = Adam(lr = lr, decay = decay)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	
	model.summary()
	return model


baselineSED = baselineSED(rows,cols,channels,nb_classes,lr)


# feature and label
X_train = np.load('../train_feature.npy')
Y_train = np.load('../train_label_SED.npy')
X_val = np.load('../val_feature.npy')
Y_val = np.load('../val_label_SED.npy')

## training
def model_train(model,X_train,Y_train,X_val,Y_val,epochs,batch_size,model_name,output_folder):
	filepath=os.path.join(output_folder,model_name + '-{epoch:02d}-{val_loss:.2f}.hdf5')
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]
	hist = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), callbacks=callbacks_list, epochs=epochs, shuffle=False,batch_size=batch_size,verbose=2)
	return hist
	
	
train_history = model_train(baselineSED,X_train,Y_train,X_val,Y_val,epochs,batch_size,SED,output_folder)
