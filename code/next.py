#!/usr/bin/python

"""Finds the proximity sensors reading for a given distance in the future."""

import os
import tqdm
import numpy as np
import pandas as pd
from datetime import timedelta


### Util


def in_range(x, range):
	"""Tells wether a certain value is inside a given range (exclusive).

	Args:
		x: a value.
		range: a tuple representing the lowerbound and upperbound of the range (exclusive).

	Returns:
		a boolean value representing that the values is inside the range.
	"""
	return np.logical_and(range[0] < x,  x < range[1])

def filter(dist, dx, dy, dtheta):
	"""Filters the dataframe using ranges for distance, delta position x and y and delta theta

	Args:
		dist: a distance expressed in cm.
		dx: an array of delta x positions expressed in cm.
		dy: an array of delta y positions expressed in cm.
		dtheta: an array of delta angles expressed in radians.

	Returns:
		a boolean array with True in the positions in which the data passed the filter.
	"""
	dist /= 100 # convert from cm to m
	return in_range(dx, (dist - 0.01, dist + 0.01)) & \
			in_range(dy, (-0.01, 0.01)) & \
			in_range(dtheta, (-0.05, 0.05))

def relative_to(position, frame):
	"""Computes the position relative to the given frame of reference.

	Args:
		position: a position expressed in 2d homogeneous coordinates (x, y, 1).
		frame: a frame of reference from which to extract pos_x, pos_y and theta.

	Returns:
		the relative position wrt the given frame.
	"""
	cos = np.cos(frame['theta'])
	sin = np.sin(frame['theta'])
	x = frame['pos_x']
	y = frame['pos_y']

	inverse_frame = np.linalg.inv(np.array([[cos, -sin, x], [sin, cos, y], [0, 0, 1]]))

	return np.matmul(inverse_frame, position)


### Code


def find_next(row, df, dist, min_speed, max_speed):
	"""Finds the proximity sensors readings for a given pose in the future.

	Args:
		row: a dataframe row containing the pose from which to compute the next pose.
		df: the dataframe from which to extract the next pose.
		dist: a distance expressed in cm.
		min_speed: the minimum thymio speed expressed in cm/s.
		max_speed: the maximum thymio speed expressed in cm/s.

	Returns:
		The proximity sensors readings for a pose in the future, if there is one, otherwise a row of None.
	"""
	idx = row.name

	window = df.loc[idx + pd.Timedelta(str(round(dist / max_speed, 1)) + 's'):
					idx + pd.Timedelta(str(round(dist / min_speed, 1)) + 's')]

	prefix = 't_%.1f_' % dist

	if len(window) == 0:
		return pd.Series({
			prefix + 'prox_lx': None,
			prefix + 'prox_cl': None,
			prefix + 'prox_cx': None,
			prefix + 'prox_cr': None,
			prefix + 'prox_rx': None
			})

	next_pose = np.concatenate([
		np.expand_dims(window['pos_x'].values, axis=1),
		np.expand_dims(window['pos_y'].values, axis=1),
		np.ones((len(window), 1))], axis=1)

	relative_pose = np.array([relative_to(next_pose[i, :], row) for i in range(len(window))])

	dx = relative_pose[:, 0]
	dy = relative_pose[:, 1]
	dtheta = (window['theta'] - row['theta']).values

	results = window[filter(dist, dx, dy, dtheta)]

	if len(results) == 0:
		return pd.Series({
			prefix + 'prox_lx': None,
			prefix + 'prox_cl': None,
			prefix + 'prox_cx': None,
			prefix + 'prox_cr': None,
			prefix + 'prox_rx': None
			})
	elif len(results) == 1:
		return pd.Series({
			prefix + 'prox_lx': results['prox_lx'][0],
			prefix + 'prox_cl': results['prox_cl'][0],
			prefix + 'prox_cx': results['prox_cx'][0],
			prefix + 'prox_cr': results['prox_cr'][0],
			prefix + 'prox_rx': results['prox_rx'][0]
			})
	else:
		return pd.Series({
			prefix + 'prox_lx': results.iloc[0]['prox_lx'],
			prefix + 'prox_cl': results.iloc[0]['prox_cl'],
			prefix + 'prox_cx': results.iloc[0]['prox_cx'],
			prefix + 'prox_cr': results.iloc[0]['prox_cr'],
			prefix + 'prox_rx': results.iloc[0]['prox_rx']
			})

def next_prox(df, distances):
	"""Finds the proximity sensors reading for a given distance in the future and saves the data in the same dataframe.

	Args:
		df: the dataframe in which to find the data to elaborate.
		distances: the different distances from which data is extracted.

	Returns:
		The dataframe with the new columns corresponding to the proximity reading for various distances.
	"""
	for dist in tqdm.tqdm(distances, desc='extracting future readings'):

		min_speed = 5 #cm/s
		max_speed = 20 #cm/s

		next_laser = df.apply(find_next, axis=1, args=(df, dist, min_speed, max_speed))

		df = pd.concat([df, next_laser], axis=1)

	return df
