#!/usr/bin/python

"""Train the neural network model using the given training set."""

import os
import keras
import argparse
import numpy as np
import pandas as pd
from model import model, generator
from datetime import datetime
import matplotlib.pyplot as plt

def train():
	"""Train the neural network model, save the weights and show the learning error over time."""
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--name', type=str, help='name of the Model weights', default='model_' + str(datetime.now()))
	parser.add_argument('-f', '--filename', type=str, help='name of the dataset (.h5 file)', default='data_gazebo.h5')
	parser.add_argument('-e', '--epochs', type=int, help='number of epochs of the training phase', default=100)
	parser.add_argument('-s', '--steps', type=int, help='number of training steps per epoch', default=1000)
	parser.add_argument('-sp', '--split-percentage', type=float, help='train/test split percentage (0:100)', default=50.0)
	parser.add_argument('-bs', '--batch-size', type=int, help='size of the batches of the training data', default=64)
	parser.add_argument('-lr', '--learning-rate', type=float, help='learning rate used for the training phase', default=0.0002)
	args = parser.parse_args()

	name = args.name
	filename = args.filename
	n_epochs = args.epochs
	steps = args.steps
	batch_size = args.batch_size
	learning_rate = args.learning_rate
	split_percentage = args.split_percentage

	os.mkdir(name)

	print()
	print('Parameters:')
	for k, v in vars(args).items():
		print(k, '=', v)
	print()

	cnn = model(learning_rate, show_summary=False)
	gen = generator(split_percentage=split_percentage, filename=filename, batch_size=batch_size,
					is_testset=False, augment=True)

	val_x, val_y, _ = next(generator(filename=filename, is_testset=True, testset_index=-1))

	history = cnn.fit_generator(generator=gen, steps_per_epoch=steps, epochs=n_epochs,
		validation_data = (val_x, val_y), callbacks=[
			keras.callbacks.ModelCheckpoint(name +'/weights.{epoch:02d}-{val_loss:.4f}.h5', save_best_only=True, save_weights_only=True),
			keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
		])

if __name__ == '__main__':
	train()
