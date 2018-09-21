#!/usr/bin/python

"""Creates and visualizes the HDF5 file content as a video."""

import os
import cv2
import h5py
import numpy as np

def draw_laser(frame, prox, bar_w, bar_h, off_x, size):
	"""Draws the sensor readings onto the image using rectangles

	Args:
		frame: a numpy array containing the raw image
		prox: a list of readings
		bar_w: the width of the rectangle
		bar_h: the height of the rectangle
		off_x: a number representing the horizontal offset
		size: the size of the frame
	"""
	maxx = 4700
	cv2.rectangle(frame,
				(off_x, size[1] - int(5 + bar_h * prox[0] / maxx)),
				(off_x + bar_w, size[1]),
				(0, int(255 - (prox[0] / maxx) * 255), 255), cv2.FILLED)
	cv2.rectangle(frame,
				(off_x + bar_w, size[1] - int(5 + bar_h * prox[1] / maxx)),
				(off_x + bar_w * 2, size[1]),
				(0, int(255 - (prox[1] / maxx) * 255), 255), cv2.FILLED)
	cv2.rectangle(frame,
				(off_x + bar_w * 2, size[1] - int(5 + bar_h * prox[2] / maxx)),
				(off_x + bar_w * 3, size[1]),
				(0, int(255 - (prox[2] / maxx) * 255), 255), cv2.FILLED)
	cv2.rectangle(frame,
				(off_x + bar_w * 3, size[1] - int(5 + bar_h * prox[3] / maxx)),
				(off_x + bar_w * 4, size[1]),
				(0, int(255 - (prox[3] / maxx) * 255), 255), cv2.FILLED)
	cv2.rectangle(frame,
				(off_x + bar_w * 4, size[1] - int(5 + bar_h * prox[4] / maxx)),
				(off_x + bar_w * 5, size[1]),
				(0, int(255 - (prox[4] / maxx) * 255), 255), cv2.FILLED)

def visualize():
	"""Creates and visualizes the HDF5 file content."""
	file = 'data/data_many_dist_fixed_step.h5'
	h5f = h5py.File(file, 'r')
	bags =  np.unique([str(b[:-2]) for b in h5f.keys()])

	for bag in bags:
		print('Found ' + bag)
		
		Xs = h5f[bag + '_x'][:]
		Ys = h5f[bag + '_y'][:]
		l = Xs.shape[0]

		size = (400, 300)
		bar_h = 80
		bar_w = 80
		off_x = 0
		video = cv2.VideoWriter(bag + '.avi', cv2.VideoWriter_fourcc(*'XVID'), 6, size)

		for i in range(l):
			x = Xs[i]
			y = Ys[i]
			
			frame = cv2.resize(x, size)
			frame = (frame - frame.min()) / (frame.max() - frame.min())
			frame = (frame * 255).astype(np.uint8)

			draw_laser(frame, y[0:5], bar_w, bar_h, off_x, size)

			video.write(frame)

			cv2.imshow(bag, frame)
			if cv2.waitKey(1000 // 10) & 0xFF == ord('q'):
				break

	video.release()
	
	cv2.destroyAllWindows()

if __name__ == '__main__':
	visualize()
