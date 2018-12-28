#!/usr/bin/python

'''Finds the proximity sensors reading for a given coordinate in the future.'''

import os
import tqdm
import numpy as np
import pandas as pd
from datetime import timedelta
from scipy.spatial import cKDTree


### Util


def relative_to(positions, frame):
	'''Computes the positions relative to the given frame of reference.

	Args:
		positions: an iterable of positions expressed in 2d homogeneous coordinates (x, y, 1).
		frame: a frame of reference from which to extract pos_x, pos_y and theta.

	Returns:
		the relative position wrt the given frame.
	'''
	
	cos = np.cos(frame['theta'])
	sin = np.sin(frame['theta'])
	x = frame['pos_x']
	y = frame['pos_y']

	inverse_frame = np.linalg.inv(np.array([[cos, -sin, x], [sin, cos, y], [0, 0, 1]]))
	return np.matmul(inverse_frame, np.array(positions).T).T


### Code


def find_next(row, df, coords, target_columns, td_wbegin, td_wend):
	'''Finds the proximity sensors readings for a given pose in the past and future.

	Args:
		row: a dataframe row containing the pose from which to compute the next pose.
		df: the dataframe from which to extract the next pose.
		coords: a list of coordinates of the form [(x, y), ...].
		target_columns: a list of columns used as the targets readings.
		td_wbegin: initial time of the window (relative to row index)
		td_wend: final time of the window (relative to row index)

	Returns:
		An iterable (one per element in coord) containing an iterable (one per target_column)
	'''
	idx = row.name
	window = df.loc[idx + td_wbegin:idx + td_wend]
	values = np.full((len(target_columns), len(coords)), -1.0)

	if len(window) == 0:
		return pd.Series({c: values[i] for i, c in enumerate(target_columns)})

	next_pose = np.concatenate([
		np.expand_dims(window['pos_x'].values, axis=1),
		np.expand_dims(window['pos_y'].values, axis=1),
		np.ones((len(window), 1))], axis=1)

	relative_pose = relative_to(next_pose, row)
	dx = relative_pose[:, 0] / relative_pose[:, 2]
	dy = relative_pose[:, 1] / relative_pose[:, 2]

	if np.any(relative_pose[:, 2] != 1.0):
		print('next.py:196: np.any(relative_pose[:, 2] != 1.0')

	kdt = cKDTree(relative_pose[:,[0, 1]])
	distances, indices = kdt.query(np.array(coords) / 100.0, 1, n_jobs=-1)
	
	# use only values from poses distant at max 10 cm from the relative coord
	ix = np.nonzero(distances < 0.1)[0]

	for i, c in enumerate(target_columns):
		values[i, ix] = window.iloc[indices[ix]][c]

	return pd.Series({c: values[i] for i, c in enumerate(target_columns)})

def next_prox(df, coords, target_columns):
	'''Finds the proximity sensors reading for a given pose and saves the data in the same dataframe.

	Args:
		df: the dataframe in which to find the data to elaborate.
		coords: a list of relative coordinates of the form [(x, y), ...].
		target_columns: a list of columns used as the targets readings.

	Returns:
		The dataframe with the new columns corresponding to the proximity reading for various distances.
	'''
	print('extracting target values...')
	
	td_wbegin = pd.Timedelta('-60 s')
	td_wend = pd.Timedelta('+60 s')
	
	next_df = df.apply(find_next, axis=1, args=(df, coords, target_columns, td_wbegin, td_wend))
	
	output_cols = next_df.columns.values.tolist()
	df = pd.concat([df.drop(target_columns, axis=1), next_df], axis=1)
	
	return df, output_cols
