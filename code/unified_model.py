#!/usr/bin/python

"""Function to create the keras model and prepare the data before the training or testing phase."""

import h5py
import numpy as np
import pandas as pd
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam

def makeRandomGradient(size):
	"""Creates a random gradient

	Args:
		size: the size of the gradient

	Returns:
		the random gradient.
	"""
	x, y = np.meshgrid(np.linspace(0, 1, size[1]), np.linspace(0, 1, size[0]))
	grad = x * np.random.uniform(-1, 1) + y * np.random.uniform(-1, 1)
	grad = (grad - grad.mean()) / grad.std()
	return grad

def alter(x):
	"""Applies a random gradient to the image

	Args:
		x: an image represented by a 3d numpy array

	Returns:
		the image with the added random gradient.
	"""
	grad = makeRandomGradient(x.shape)

	for i in range(3):
		x[:, :, i] = x[:, :, i] * np.random.uniform(0.9, 1.1)
	x = (x - x.mean()) / x.std()

	amount = np.random.uniform(0.05, 0.15)

	for i in range(3):
		x[:, :, i] = x[:, :, i] * (1 - amount) + grad * amount
	x = (x - x.mean()) / x.std()

	return x

def additive_noise(x):
	"""Adds gaussian noise centered on 0 to an image.

	Args:
		x: an image represented by a 3d numpy array

	Returns:
		the image with the added noise.
	"""
	gauss = np.random.normal(0, 2 * 1e-2, x.shape) # 2% gaussian noise
	x = x + gauss
	return x

def grayscale(x):
	"""Converts an image to grayscale.

	Args:
		x: an image represented by a 3d numpy array

	Returns:
		the grayscale image.
	"""
	return np.dstack([0.21 * x[:,:,2] + 0.72 * x[:,:,1] + 0.07 * x[:,:,0]] * 3)

def flip(x, y):
	"""Flips an image and the corresponding labels.

	Args:
		x: an image represented by a 3d numpy array
		y: a list of labels associated with the image

	Returns:
		the flipped image and labels.
	"""
	if np.random.choice([True, False]):
		x = np.fliplr(x)

		for i in range(len(y) // 5):
			y[i * 5:(i + 1) * 5] = np.flipud(y[i * 5:(i + 1) * 5])

	return (x, y)

def random_augment(im):
	choice = np.random.randint(0, 3)

	if choice == 0:
		im = additive_noise(im)
	elif choice == 1:
		im = grayscale(im)

	im = (im - im.mean()) / im.std()

	im = alter(im)

	return im

def generator(group, batch_size, is_testset=False, augment=True, do_flip=True):
	"""Loads the dataset, preprocess it and generates batches of data.

	Args:
		group: list of ids from which to generate the data.
		batch_size: the size of the batch.
		is_testset: a boolean flag representing if the genrated data will be used as test-set.
		augment: a boolean flag representing wether to augment the data or not.
		do_flip: a boolean flag representing wether to also flip horizontally the images and the labels.

	Returns:
		the preprocessed batches.
	"""
	h5f = h5py.File('data/data_many_dist_fixed_step.h5', 'r')

	
	Xs = {i: h5f['bag' + str(i) +'_x'] for i in group}
	Ys = {i: h5f['bag' + str(i) +'_y'] for i in group}
	lengths = {i: Xs[i].shape[0] for i in group}
	counts = {i: 0 for i in group}

	if is_testset and len(group) == 1:
		x = Xs[group[0]][:]
		y = Ys[group[0]][:]

		if do_flip:
			for i in range(x.shape[0]):
				x[i], y[i] = flip(x[i], y[i])

		y[y > 0] = 1.0
		mask = y != -1.0

		yield (x, y, mask)

	else:
		while True:
			index = np.random.choice(group)
			
			x = Xs[index][counts[index]:counts[index] + batch_size]
			y = Ys[index][counts[index]:counts[index] + batch_size]
			
			counts[index] += batch_size
			
			if counts[index] + batch_size > lengths[index]:
				counts[index] = 0

			if augment:
				for i in range(x.shape[0]):
					x[i] = random_augment(x[i])

			if do_flip:
				for i in range(x.shape[0]):
					x[i], y[i] = flip(x[i], y[i])

			y[y > 0] = 1.0 # binary classes 0, 1 and -1 for missing value

			yield (x, y)

def model(lr=0.001, show_summary=False, old_version=True):
	"""Creates the keras neural network model.

	Args:
		lr: the learning rate used for the training.
		show_summary: a boolean flag that represents if the model has to be printed to console.
		old_version: a boolean flag that represents if the model should be the old one or the new one (more neurons).

	Returns:
		The defined keras model.
	"""
	model = Sequential()

	model.add(Conv2D(10, (3, 3), padding='same', input_shape=(64, 80, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(10, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(8, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())

	if old_version == True:
		model.add(Dense(32))
	else:
		model.add(Dropout(0.2))
		model.add(Dense(256))

	model.add(Activation('relu'))
	if old_version:
		model.add(Dense(20))
	else:
		model.add(Dense(65 * 5))
	model.add(Activation('sigmoid', name='output'))

	def masked_mse(target, pred):
		mask = K.cast(K.not_equal(target, -1), K.floatx())
		mse = K.mean(K.square((pred - target) * mask))
		return mse

	model.compile(loss=masked_mse, optimizer=Adam(lr=lr))

	if show_summary:
		model.summary()

	return model
