#!/usr/bin/python

"""Train the neural network model using the given training set."""

import argparse
import numpy as np
import pandas as pd
from unified_model import model, generator
from datetime import datetime
import matplotlib.pyplot as plt

def train():
	"""Train the neural network model, save the weights and shows the learning error over time."""
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--name', type=str, help='name of the Model weights', default='model_' + str(datetime.now()))
	parser.add_argument('-e', '--epochs', type=int, help='number of epochs of the training phase', default=60)
	parser.add_argument('-s', '--steps', type=int, help='number of training steps per epoch', default=1000)
	parser.add_argument('-bs', '--batch-size', type=int, help='size of the batches of the training data', default=64)
	parser.add_argument('-lr', '--learning-rate', type=float, help='learning rate used for the training phase', default=0.0002)
	args = parser.parse_args()

	name = args.name
	n_epochs = args.epochs
	steps = args.steps
	batch_size = args.batch_size
	learning_rate = args.learning_rate

	print()
	print('Parameters:')
	for k, v in vars(args).items():
		print(k, '=', v)
	print()

	cnn = model(learning_rate, show_summary=True, old_version=False)
	gen = generator(np.arange(0, 9), batch_size, is_testset=False, augment=True, do_flip=True)

	validation = next(generator(np.arange(9, 11), 1000, is_testset=False, augment=True, do_flip=True))

	history = cnn.fit_generator(generator=gen, steps_per_epoch=steps, epochs=n_epochs,
		validation_data = validation)

	filename = 'model/' + name + '.h5'
	cnn.save_weights(filename)

	l = len(history.history)

	for i, t in enumerate(history.history.items()):
		plt.subplot(1, l, i + 1)
		plt.plot(t[1])
		plt.title(t[0])
		print(t[0])
		print(t[1])
		print('-------------------')

	plt.show()

if __name__ == '__main__':
	train()
