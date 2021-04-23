
# define cnn model
import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


# define cnn model
def define_model():
	# # VGG1
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

	# VGG2
	# model = Sequential()
	# model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	# model.add(MaxPooling2D((2, 2)))
	# model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	# model.add(MaxPooling2D((2, 2)))
	# model.add(Flatten())
	# model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	# model.add(Dense(1, activation='sigmoid'))
	# # compile model
	# opt = SGD(lr=0.001, momentum=0.9)
	# model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	# return model

	# Transfer learning
	# model = VGG16(include_top=False, input_shape=(224, 224, 3))
	# # mark loaded layers as not trainable
	# for layer in model.layers:
	# 	layer.trainable = False
	# # add new classifier layers
	# flat1 = Flatten()(model.layers[-1].output)
	# class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
	# output = Dense(1, activation='sigmoid')(class1)
	# # define new model
	# model = Model(inputs=model.inputs, outputs=output)
	# # compile model
	# opt = SGD(lr=0.001, momentum=0.9)
	# model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	# return model



# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	pyplot.savefig('vgg1')
	# pyplot.savefig('vgg2' )
	# pyplot.savefig('transfer_learning' )
	pyplot.close()

# run the test harness for evaluating a model
def run_test_harness():
	# define model
	model = define_model()
	# create data generator
	datagen = ImageDataGenerator(rescale=1.0/255.0)
	# prepare iterators
	train_it = datagen.flow_from_directory('images/Train/',
		class_mode='binary', target_size=(200, 200))
	test_it = datagen.flow_from_directory('images/Test/',
		class_mode='binary', target_size=(200, 200))
	# fit model
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
		validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)
	# evaluate model
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history)

# entry point, run the test harness
run_test_harness()