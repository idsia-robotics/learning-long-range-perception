#!/usr/bin/python

'''Finds the proximity sensors reading for a given coordinate in the future.'''

import os
import tqdm
import numpy as np
import pandas as pd
from datetime import timedelta
from scipy.spatial import cKDTree


### Util


# def in_range(x, range):
# 	'''Tells wether a certain value is inside a given range (exclusive).

# 	Args:
# 		x: a value.
# 		range: a tuple representing the lowerbound and upperbound of the range (exclusive).

# 	Returns:
# 		a boolean value representing that the values is inside the range.
# 	'''
# 	return np.logical_and(range[0] < x,  x < range[1])

# def filter(coord, dx, dy, dtheta):
# 	'''Filters the dataframe using ranges for each component of the x,y coordinate and the angle theta

# 	Args:
# 		coord: a coordinate of the form (x, y, theta).
# 		dx: an array of delta x positions expressed in cm.
# 		dy: an array of delta y positions expressed in cm.
# 		dtheta: an array of delta angles expressed in radians.

# 	Returns:
# 		a boolean array with True in the positions in which the data passed the filter.
# 	'''
# 	cx = coord[0] / 100.0 # convert from cm to m
# 	cy = coord[1] / 100.0 # convert from cm to m
# 	return in_range(dx, (cx - 0.05, cx + 0.05)) & \
# 			in_range(dy, (cy - 0.05, cy + 0.05))

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


# def find_next_olde(row, df, coord, target_columns, min_speed, max_speed):
# 	'''Finds the proximity sensors readings for a given pose in the past and future.

# 	Args:
# 		row: a dataframe row containing the pose from which to compute the next pose.
# 		df: the dataframe from which to extract the next pose.
# 		coord: a list of coordinates of the form [(x, y, theta), ...].
# 		target_columns: a list of columns used as the targets readings.
# 		min_speed: the minimum thymio speed expressed in cm/s.
# 		max_speed: the maximum thymio speed expressed in cm/s.

# 	Returns:
# 		The proximity sensors readings for a pose in the future, if there is one, otherwise a row of None.
# 	'''
# 	# for coord in coords:
# 	idx = row.name
# 	dist = np.sqrt(coord[0] ** 2 + coord[1] ** 2)
# 	coord_str = str(coord[0]) + '_' + str(coord[1])
	
# 	window = pd.concat([df.loc[idx - pd.Timedelta(str(round(dist / min_speed, 1)) + 's'):
# 							   idx - pd.Timedelta(str(round(dist / max_speed, 1)) + 's')],
# 						df.loc[idx + pd.Timedelta(str(round(dist / max_speed, 1)) + 's'):
# 							   idx + pd.Timedelta(str(round(dist / min_speed, 1)) + 's')]
# 						])

# 	if len(window) == 0:
# 		return pd.Series({c + '_' + coord_str: None for c in target_columns})

# 	next_pose = np.concatenate([
# 		np.expand_dims(window['pos_x'].values, axis=1),
# 		np.expand_dims(window['pos_y'].values, axis=1),
# 		np.ones((len(window), 1))], axis=1)

# 	relative_pose = relative_to(next_pose, row)
# 	dx = relative_pose[:, 0]
# 	dy = relative_pose[:, 1]
# 	dtheta = (window['theta'] - row['theta']).values

# 	results = window[filter(coord, dx, dy, dtheta)]

# 	values = []
# 	for c in target_columns:
# 		if len(results) == 0:
# 			values.append(None)
# 		elif len(results) == 1:
# 			values.append(results[c][0])
# 		else:
# 			values.append(results.iloc[0][c])

# 	return pd.Series({c + '_' + coord_str: v for c, v in zip(target_columns, values)})


# def find_next(row, df, coords, coord_dist, target_columns, td_wbegin, td_wend):
# 	'''Finds the proximity sensors readings for a given pose in the past and future.

# 	Args:
# 		row: a dataframe row containing the pose from which to compute the next pose.
# 		df: the dataframe from which to extract the next pose.
# 		coords: a list of coordinates of the form [(x, y, theta), ...].
# 		target_columns: a list of columns used as the targets readings.
# 		td_wbegin: initial time of the window (relative to row index)
# 		td_wend: final time of the window (relative to row index)

# 	Returns:
# 		An iterable (one per element in coord) containing an iterable (one per target_column)
# 	'''
# 	idx = row.name
# 	n_coords = len(coords)
# 	window = df.loc[idx + td_wbegin:idx + td_wend]
# 	values = np.full((len(target_columns), n_coords), -1.0)

# 	if len(window) == 0:
# 		return pd.Series({c: values[i] for i, c in enumerate(target_columns)})

# 	next_pose = np.concatenate([
# 		np.expand_dims(window['pos_x'].values, axis=1),
# 		np.expand_dims(window['pos_y'].values, axis=1),
# 		np.ones((len(window), 1))], axis=1)

# 	relative_pose = relative_to(next_pose, row)
# 	dx = relative_pose[:, 0]
# 	dy = relative_pose[:, 1]
	
# 	kdt = cKDTree(relative_pose[:,[0, 1]])
# 	distances, indices = kdt.query(coords, 1, n_jobs=-1)
# 	distances = np.abs(distances - coord_dist)
	
# 	for i, c in enumerate(target_columns):
# 		for j in range(n_coords):
# 			if distances[j] < 2.0:
# 				values[i, j] = window.iloc[indices[j]][c]

# 	return pd.Series({c: values[i] for i, c in enumerate(target_columns)})

def find_next2(row, df, coords, target_columns, td_wbegin, td_wend):
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
	
	# use only values from poses distant at max 2 cm from the relative coord
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
	
	next_df = df.apply(find_next2, axis=1, args=(df, coords, target_columns, td_wbegin, td_wend))
	
	output_cols = next_df.columns.values.tolist()
	df = pd.concat([df.drop(target_columns, axis=1), next_df], axis=1)
	
	return df, output_cols


# def gp(a): return a[['pos_x', 'pos_y'].values
# def dist(a,b): return np.linalg.norm(b-a)
# for a,b in zip(np.array(coords)[ix], (window[['pos_x','pos_y']].iloc[indices[ix]].values - row[['pos_x', 'pos_y']].values) * 100):print(np.linalg.norm(b-a))
