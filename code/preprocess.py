#!/usr/bin/python

"""Preprocess the bagfiles in a folder, synchronize the different topics and generate a single HDF5 binary file."""

import os
import cv2
import tqdm
import h5py
import rosbag
from next import *
import numpy as np
import pandas as pd
from datetime import datetime
from tf.transformations import euler_from_quaternion


### Util


def jpeg2np(image, size=None):
	"""Converts a jpeg image in a 2d numpy array of BGR pixels and resizes it to the given size (if provided).

	Args:
		image: a compressed BGR jpeg image.
		size: a tuple containing width and height, or None for no resizing.

	Returns:
		the raw, resized image as a 2d numpy array of BGR pixels.
	"""
	compressed = np.fromstring(image, np.uint8)
	raw = cv2.imdecode(compressed, cv2.IMREAD_COLOR)

	if size:
		raw = cv2.resize(raw, size)

	return raw

def quaternion2yaw(q):
	"""Converts a quaternion into the respective z euler angle.

	Args:
		q: a quaternion, composed of X, Y, Z, W.

	Returns:
		 The euler angle part for the z axis, also called yaw.
	"""

	return euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

def lerp(x, y, a):
	"""Linear interpolation between two points by a fixed amount.

	Args:
		x: the first point.
		y: the second point.
		a: the percentage between the two points.

	Returns:
		the interpolated point.
	"""
	return (1 - a) * x + a * y


### Code


def preprocess(path='./', new_thymio=False, distances=None):
	"""Extracts data to bagfiles in a given path and converts them into dataframes and saves them into an HDF5 binary file.

	Args:
		path: the path in which bag files are.
		new_thymio: a string containing the number of the thymio for new thymios or False for the old version thymio.
	"""
	if new_thymio:
		cam = '/thymio' + new_thymio + '/camera/image_raw/compressed'
		prox = '/thymio' + new_thymio + '/proximity/laser'
		odom = '/thymio' + new_thymio + '/odom'		
	else:
		cam = '/camera/image_rect_color/compressed'
		prox = '/proximity/laser'
		odom = '/odom'

	if distances is None:
		distances = [0.0, 3.3, 6.6, 10.0]

	extractors = {
		cam: lambda m: {
						'camera': jpeg2np(m.data, (80, 64))
														  },
		prox: lambda m: {
							'prox_lx': m.intensities[4],
							'prox_cl': m.intensities[3],
							'prox_cx': m.intensities[2],
							'prox_cr': m.intensities[1],
							'prox_rx': m.intensities[0]
							},
		odom: lambda m: {
							'pos_x': m.pose.pose.position.x,
							'pos_y': m.pose.pose.position.y,
							'theta': quaternion2yaw(m.pose.pose.orientation)
							}
	}

	files = [file[:-4] for file in os.listdir(path) if file[-4:] == '.bag']

	if not files:
		print('No bag files found!')
		return None

	h5f = h5py.File('data/' + str(datetime.now()) + '.h5', 'w')
	
	for index, file in enumerate(files):
		print('Found ' + path + file + '.bag')

		dfs = bag2dfs(rosbag.Bag(path + file + '.bag'), extractors)

		df = mergedfs(dfs)
		df = next_prox(df, distances)
		
		l = len(df)

		df.fillna(-1.0, inplace=True)
		df['camera'] = df['camera'].apply(lambda x: (x - x.mean()) / x.std())

		Xs = h5f.create_dataset('bag' + str(index) + '_x', shape=(l, 64, 80, 3), maxshape=(None, 64, 80, 3), dtype=np.float, data=None, chunks=True)
		Ys = h5f.create_dataset('bag' + str(index) + '_y', shape=(l, 5 * len(distances)), maxshape=(None, 5 * len(distances)), dtype=np.float, data=None, chunks=True)
		
		cols = ['t_%.1f_prox_%s' % (d, sensor) for d in distances for sensor in ['lx', 'cl', 'cx', 'cr', 'rx']]

		Xs[:] = np.stack(df['camera'].values)
		Ys[:] = df[cols].values

	h5f.close()

def bag2dfs(bag, extractors):
	"""Extracts data from a bagfile and converts it to dataframes (one per topic).

	Args:
		bag: an opened bag file.
		extractors: a dictionary of functions associated to ros topics that extracts the required values,
					composed of ros topics as keys and functions as values.
	Returns:
		a dictionary of dataframes divided by ros topic.
	"""
	result = {}

	for topic in extractors.keys():
		timestamps = []
		values = []

		for subtopic, msg, t in bag.read_messages(topic):
			if subtopic == topic:
				timestamps.append(msg.header.stamp.to_nsec())
				values.append(extractors[topic](msg))

		df = pd.DataFrame(data=values, index=timestamps, columns=values[0].keys())
		result[topic] = df

	return result

def mergedfs(dfs):
	"""Merges different dataframes into a single synchronized dataframe.

	Args:
		dfs: a dictionary of dataframes divided by ros topic.

	Returns:
		a single dataframe composed of the various dataframes synchronized.
	"""
	min_topic = None

	for topic, df in dfs.items():
		if not min_topic or len(dfs[min_topic]) > len(df):
			min_topic = topic

	values = []
	ref_df = dfs[min_topic]
	topics = dfs.keys()
	topics.remove(min_topic)
	topics = zip([0] * len(topics), topics)

	for i in tqdm.tqdm(range(0, len(ref_df)), desc='generating datapoints'):
		t = ref_df.index[i]
		row =[{'timestamp': t}, ref_df.loc[t].to_dict()]

		for idx, topic in topics:
			df = dfs[topic]
			
			while idx < len(df) and df.index[idx] < t:
				idx += 1
			
			if idx >= len(df):
				row = None
				break

			row.append(df.iloc[idx].to_dict())

		if row:
			values.append({k: v for d in row for k, v in d.items()})

	result = pd.DataFrame.from_dict(values).set_index('timestamp')
	result.index = pd.to_datetime(result.index)
	
	return result

if __name__ == '__main__':
	path = '/path/to/the/bagfiles'
	thymio = '21'
	
	distances = list(range(0, 31, 1))

	res = preprocess(path, thymio, np.array(distances, dtype=np.float))
	print('Finished')
