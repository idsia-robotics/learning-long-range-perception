#!/usr/bin/python

'''Preprocess the bagfiles in a folder, synchronize the different topics and generate a single HDF5 binary file.'''

import os
import cv2
import tqdm
import h5py
import rosbag
import numpy as np
from next import *
from settings import *
from datetime import datetime
from tf.transformations import euler_from_quaternion


### Util


def jpeg2np(image, size=None, normalize=False):
	'''Converts a jpeg image in a 2d numpy array of BGR pixels and resizes it to the given size (if provided).

	Args:
		image: a compressed BGR jpeg image.
		size: a tuple containing width and height, or None for no resizing.
		normalize: a boolean flag representing wether or not to normalize the image.

	Returns:
		the raw, resized image as a 2d numpy array of BGR pixels.
	'''
	compressed = np.fromstring(image, np.uint8)
	raw = cv2.imdecode(compressed, cv2.IMREAD_COLOR)

	if size:
		raw = cv2.resize(raw, size)

	if normalize:
		raw = (raw - raw.mean()) / raw.std()

	return raw

def quaternion2yaw(q):
	'''Converts a quaternion into the respective z euler angle.

	Args:
		q: a quaternion, composed of X, Y, Z, W.

	Returns:
		 The euler angle part for the z axis, also called yaw.
	'''

	return euler_from_quaternion([q.x, q.y, q.z, q.w])[2]


### Code


def preprocess(path='./', extractors={}, coords=[(0, 0)]):
	'''Extracts data to bagfiles in a given path and converts them into dataframes and saves them into an HDF5 binary file.

	Args:
		path: the path in which bag files are.
		extractors: a dictionary of functions associated to ros topics that extracts the required values,
					composed of ros topics as keys and functions as values.
		coords: a list of coordinates (x, y).
	'''
	files = [file[:-4] for file in os.listdir(path) if file[-4:] == '.bag']

	if not files:
		print('No bag files found!')
		return None

	h5f = h5py.File(str(datetime.now()) + '.h5', 'w')
	
	for index, file in enumerate(files):
		print('Found ' + path + file + '.bag')

		dfs = bag2dfs(rosbag.Bag(path + file + '.bag'), extractors)

		df = mergedfs(dfs)

		target_cols = [col for col in df.columns.values if 'target' in col]
		input_cols = [col for  col in df.columns.values if col not in target_cols]

		df, output_cols = next_prox(df, coords, target_cols)
		df.fillna(-1.0, inplace=True)

		l = len(df)

		print('saving dataframe...')

		for col in input_cols:
			shape = df[col].iloc[0].shape
			store = h5f.create_dataset('bag' + str(index) + '/x/' + col,
						shape=(l,) + shape, maxshape=(None,) + shape,
						dtype=np.float, data=None, chunks=True)
			store[:] = np.stack(df[col].values)

		for col in output_cols:
			shape = df[col].iloc[0].shape
			store = h5f.create_dataset('bag' + str(index) + '/y/' + col,
						shape=(l,) + shape, maxshape=(None,) + shape,
						dtype=np.float, data=None, chunks=True)
			store[:] = np.stack(df[col].values)

	h5f.close()

def bag2dfs(bag, extractors):
	'''Extracts data from a bagfile and converts it to dataframes (one per topic).

	Args:
		bag: an opened bag file.
		extractors: a dictionary of functions associated to ros topics that extracts the required values,
					composed of ros topics as keys and functions as values.
	Returns:
		a dictionary of dataframes divided by ros topic.
	'''
	result = {}

	# for topic in extractors.keys():
	for topic in tqdm.tqdm(extractors.keys(), desc='extracting data from the bagfile'):
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
	'''Merges different dataframes into a single synchronized dataframe.

	Args:
		dfs: a dictionary of dataframes divided by ros topic.
	
	Returns:
		a single dataframe composed of the various dataframes synchronized.
	'''
	min_topic = None

	for topic, df in dfs.items():
		if not min_topic or len(dfs[min_topic]) > len(df):
			min_topic = topic

	ref_df = dfs[min_topic]
	other_dfs = dfs
	other_dfs.pop(min_topic)

	result = pd.concat(
		[ref_df] +
		[df.reindex(index=ref_df.index, method='nearest', tolerance=pd.Timedelta('0.5s').value) for _, df in other_dfs.items()],
		axis=1)
	
	result.dropna(inplace=True)
	result.index = pd.to_datetime(result.index)
	
	return result

if __name__ == '__main__':
	path = './'

	prefix = '/pioneer3at'

	extractors = {
		prefix + '/camera_one/image_raw/compressed': lambda m: {
						'cam1': jpeg2np(m.data, (80, 64), True)
														  },
		prefix + '/camera_two/image_raw/compressed': lambda m: {
						'cam2': jpeg2np(m.data, (80, 64), True)
														  },
		prefix + '/camera_three/image_raw/compressed': lambda m: {
						'cam3': jpeg2np(m.data, (80, 64), True)
														  },
		prefix + '/camera_down/image_raw/compressed': lambda m: {
						'target1': np.mean(jpeg2np(m.data))
														  },
		prefix + '/odom': lambda m: {
							'pos_x': m.pose.pose.position.x,
							'pos_y': m.pose.pose.position.y,
							'theta': quaternion2yaw(m.pose.pose.orientation)
							}
	}

	res = preprocess(path, extractors, coords)
	print('Finished')
