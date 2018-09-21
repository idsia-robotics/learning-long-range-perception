#!/usr/bin/python

"""Test the neural network model using the test set."""

import os
import tqdm
import pickle
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from unified_model import *
from sklearn.metrics import roc_auc_score, roc_curve

def test():
	"""Test the neural network model with the given weights and outputs metrics."""
	
	distances = list(range(0, 31, 1))

	rounds = 100
	print('number of rounds = %d' % rounds)

	if not distances:
		distances = [0.0, 3.3, 6.6, 10.0]	

	n_dist = len(distances)

	directions = ['lx', 'cl', 'cx', 'cr', 'rx']
	auc_array = []
	fpr = dict()
	tpr = dict()

	cnn = model(show_summary=True, old_version=False)

	files = [file for file in os.listdir('model/') if file[-3:] == '.h5']

	print('Found models:')
	for i in range(len(files)):
		print('\t', i, ': ', files[i])

	model_index = input('Please insert the model index: ')

	cnn.load_weights('model/' + files[int(model_index)])

	rng = np.random.RandomState(13) # 13 lucky number

	for index in [9, 10]:
		print('dataset ' + str(index))
		group = [index]

		test_x, test_y, mask = next(generator(group, 32, is_testset=True, augment=False, do_flip=True))
		loss = cnn.evaluate(test_x, test_y, verbose=1)
		print('Test loss:', loss)
		prediction = cnn.predict(test_x)
		
		for p in tqdm.tqdm(range(rounds)):
			aucs = np.zeros([n_dist, 5])
			for i, d in enumerate(distances):
				d = '%.1f' % d
				for j in range(5):
					indices = np.where(mask[:, i * 5 + j])
					if len(indices[0]) > 0:
						indices = (rng.choice(indices[0], len(indices[0])), )
					try:
						auc = roc_auc_score(test_y[indices, i * 5 + j].tolist()[0], prediction[indices, i * 5 + j].tolist()[0])
					except ValueError as e:
						# print('For distance %s cm only one class is present -> auc = 0.5' % d)
						auc = 0.5
					aucs[n_dist - 1 - i, j] = auc

			auc_array.append(aucs)

	mean_auc = np.mean(auc_array, axis=0)
	std_auc = np.std(auc_array, axis=0)

	# AUC MAP

	dist_labels = ['%.0f' % d for d in np.flipud(distances)]
	dist_labels = [d for i, d in enumerate(dist_labels) if i % 2 == 0]
	colors = ['aqua', 'darkorange', 'deeppink', 'cornflowerblue', 'green']

	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.75, 3.8))
	sns.heatmap(mean_auc * 100, cmap='gray', annot=True, vmin=50, vmax=100, annot_kws={"color":"white"})
	ax.set_xticklabels(['left', '', 'center', '', 'right'])
	ax.set_yticklabels(dist_labels, rotation=0)
	plt.xlabel('Sensors')
	plt.ylabel('Distance [cm]')
	plt.title('Area Under the pReceiver Operating Characteristic Curve')
	plt.show()

	# AUC vs DISTANCE

	# todo: use percentile 2.5 and 97.5 instead of mean +/- std

	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.75, 3))
	plt.plot(distances, np.flipud(mean_auc[:, 0]), color='#3333CC', lw=2, label='lateral')
	
	ax.fill_between(distances, np.flipud(mean_auc[:, 0]) + np.flipud(std_auc[:, 0]),
							   np.flipud(mean_auc[:, 0]) - np.flipud(std_auc[:, 0]),
							   color="#3333CC", edgecolor="", alpha=0.35)

	plt.plot(distances, np.flipud(mean_auc[:, 2]), color='#000000', lw=2, label='central')
	ax.fill_between(distances, np.flipud(mean_auc[:, 2]) + np.flipud(std_auc[:, 2]),
							   np.flipud(mean_auc[:, 2]) - np.flipud(std_auc[:, 2]),
							   color="#000000", edgecolor="", alpha=0.35)

	plt.xticks(np.flipud(distances), dist_labels, rotation=90)
	plt.ylabel('AUC')
	plt.ylim(ymax=1.0)
	plt.xlabel('Distance [cm]')
	plt.title('AUC vs distance')
	plt.legend(loc="upper right")
	plt.show()

if __name__ == '__main__':
	test()
