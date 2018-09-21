#!/usr/bin/python

"""Visualize the content of the HDF5 file along with the prediction made by the model."""

import os
import cv2
import numpy as np
import pandas as pd
from unified_model import *
import matplotlib.pyplot as plt

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

def draw_laser(frame, offset, spacing, height, width, readings, is_pred):
	"""Draws the model prediction and the sensors onto the image using rectangles

	Args:
		frame: a numpy array containing the raw image
		offset: a tuple containing vertical and horizontal offset
		spacing: a number representing the spacing between each sensor rectangle
		height: the height of the rectangle
		width: the width of the rectangle
		readings: a list of readings
		is_pred: boolean flag representing if readings are taken from the model or from the sensors.
	"""
	if is_pred:
		a = np.array([230, 230, 255])
		b = np.array([25, 25, 150])
	else:
		readings = (readings > 0).astype(np.float)
		a = np.array([255, 230, 230])
		b = np.array([150, 50, 50])

	for i in range(len(readings)):
		cv2.rectangle(frame,
					(offset[0] + spacing + (spacing + width) * i, offset[1]),
					(offset[0] + width + spacing + (spacing + width) * i, offset[1] + height),
					lerp(a, b, readings[i]).tolist(), cv2.FILLED)

def visualize_output():
	"""Visualize the content of the HDF5 file along with the prediction made by the model."""
	bag_index = 0

	x, y, _ = next(generator([bag_index], 32, is_testset=True, augment=False, do_flip=False))
	l = x.shape[0]
	y = y.reshape([l, -1, 5])[:, :31, :]
	d = y.shape[1]

	print('Generating predictions...')
	
	cnn = model(old_version=False)
	cnn.load_weights('model/model_' + 'icra_many_dist_fixed_step' + '.h5')

	pred = cnn.predict(x)

	spacing = 5
	height = int(np.floor(200.0 / d))
	width = 30
	video = cv2.VideoWriter('video' + str(bag_index) + '.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (400, 300 + 220))

	print('Making the video...')

	for i in range(l):
		frame = cv2.resize(x[i], (400, 300))
		frame = (frame - frame.min()) / (frame.max() - frame.min())
		frame = (frame * 255).astype(np.uint8)
		frame = np.vstack([np.ones((220, 400, 3), np.uint8) * 255, frame])

		for j in range(d):
			draw_laser(frame, (0, 200 - j * height), spacing, height, width, pred[i, j * 5:(j + 1) * 5], True)

			if y[i, j, 0] != -1:
				draw_laser(frame, (220, 200 - j * height), spacing, height, width, y[i, j, :], False)
			else:
				pass
				cv2.rectangle(frame,
					(225, 205 - j * height),
					(395, 205 - (j + 1) * height),
					(128, 128, 128), cv2.FILLED)

		video.write(frame)

		cv2.imshow('frame', frame)
		if cv2.waitKey(1000 // 10) & 0xFF == ord('q'):
			break

	video.release()

	cv2.destroyAllWindows()

if __name__ == '__main__':
	visualize_output()
