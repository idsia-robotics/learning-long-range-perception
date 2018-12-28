#!/usr/bin/python

"""Function to create the keras model and prepare the data before the training or testing phase."""

import h5py
import numpy as np
import pandas as pd
from settings import *
from keras import backend as K
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Concatenate, Input
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

def generator(split_percentage=50.0, filename=None, batch_size=1, augment=True, is_testset=False, testset_index=0):
	"""Loads the dataset, preprocess it and generates batches of data.

	Args:
		split_percentage: a percentage (from 0 to 100) representing the split between training and testing sets.
		filename: a filename for an hdf5 storage.
		batch_size: the size of the batch.
		augment: a boolean flag representing wether to augment the data or not.
		is_testset: a boolean flag representing wether to generate data for the testing or training.
		testset_index: the index of the hdf5 storage data to be used as testset.

	Returns:
		the preprocessed batches.
	"""
	if filename is None:
		print('Error: filename for the generator is not set')
		exit()

	h5f = h5py.File(filename, 'r')

	n_bags = len(h5f.keys())
	bag_indices = np.arange(0, n_bags)

	Xkeys = [k for k in h5f['bag0/x'].keys() if k not in ['pos_x', 'pos_y', 'theta']]
	Ykeys = [k for k in h5f['bag0/y'].keys()]

	Xs = {i: {k: h5f['bag' + str(i) + '/x/' + k] for k in Xkeys} for i in bag_indices}
	Ys = {i: {k: h5f['bag' + str(i) + '/y/' + k] for k in Ykeys} for i in bag_indices}
	lengths = {i: Xs[i][Xkeys[0]].shape[0] for i in bag_indices}
	counts = {i: 0 for i in bag_indices}

	if is_testset:
		index = n_bags -1 if testset_index == -1 else testset_index

		inputs = {'input_' + k: Xs[index][k][:] for k in Xkeys}
		outputs = {'output_' + k: Ys[index][k][:] for k in Ykeys}
		masks = {'mask_' + k: Ys[index][k][:] != -1.0 for k in Ykeys}

		for k in Ykeys:
			out = outputs['output_' + k]
			out[(0 <= out) & (out <= 128)] = 1.0
			out[out > 128] = 0.0
		
		yield (inputs, outputs, masks)

	else:
		group = bag_indices[:int(n_bags * split_percentage / 100.0)]
		
		while True:
			index = np.random.choice(group)

			inputs = {'input_' + k: Xs[index][k][counts[index]:counts[index] + batch_size] for k in Xkeys}
			outputs = {'output_' + k: Ys[index][k][counts[index]:counts[index] + batch_size] for k in Ykeys}
			masks = {'mask_' + k: Ys[index][k][counts[index]:counts[index] + batch_size] != -1.0 for k in Ykeys}
			
			counts[index] += batch_size
			
			if counts[index] + batch_size > lengths[index]:
				counts[index] = 0

			if augment:
				for k in Xkeys:
					inp = inputs['input_' + k]
					for i in range(batch_size):
						inp[i] = random_augment(inp[i])	

			for k in Ykeys:
				out = outputs['output_' + k]
				out[(0 <= out) & (out <= 128)] = 1.0
				out[out > 128] = 0.0

			yield (inputs, outputs)

def model(lr=0.001, show_summary=False):
	"""Creates the keras neural network model.

	Args:
		lr: the learning rate used for the training.
		show_summary: a boolean flag that represents if the model has to be printed to console.
		
	Returns:
		The defined keras model.
	"""	
	input_cam1 = Input(shape=(64, 80, 3), name='input_cam1')
	input_cam2 = Input(shape=(64, 80, 3), name='input_cam2')
	input_cam3 = Input(shape=(64, 80, 3), name='input_cam3')

	inputs = [input_cam1, input_cam2, input_cam3]

	conv_inputs = Concatenate(axis=-1)([input_cam1, input_cam2, input_cam3])

	def conv2d(inp, filters):
		result = Conv2D(filters, (3, 3), padding='same', activation='relu')(inp)
		result = MaxPooling2D(pool_size=(2, 2))(result)
		return result

	conv_part = conv2d(conv_inputs, 20)
	conv_part = conv2d(conv_part, 12)
	conv_part = conv2d(conv_part, 10)
	conv_part = conv2d(conv_part, 8)

	ff_part = Flatten()(conv_part)
	ff_part = Dense(512, activation='relu')(ff_part)

	outputs = []
	target_columns = ['target1']
	for label in target_columns:
		outputs.append(Dense(len(coords), activation='sigmoid', name='output_' + label)(ff_part))

	model = Model(inputs=inputs, outputs=outputs)

	def masked_mse(target, pred):
		mask = K.cast(K.not_equal(target, -1), K.floatx())
		mse = K.mean(K.square((pred - target) * mask))
		return mse

	model.compile(loss={'output_' + label: masked_mse for label in target_columns}, optimizer=Adam(lr=lr))

	if show_summary:
		model.summary()

	return model
