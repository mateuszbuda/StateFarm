__author__ = 'mateusz'

import argparse
import glob
import caffe
import numpy as np
from scipy.misc import imread, imresize
import os
import pandas as pd
import time


# FIXME: duplicated code from classify.py and extract_features.py alert!


def main(net, imagesDir):
	if imagesDir[-1] != '/':
		imagesDir += '/'

	filenames = glob.glob(imagesDir + '*.jpg')
	predictions = batch_predict(net, filenames)

	makeSubmission(predictions, filenames)


def batch_predict(net, filenames):
	"""
	Get the features for all images from filenames using a network

	Inputs:
	filenames: a list of names of image files

	Returns:
	an array of feature vectors for the images in that file
	"""

	N, C, H, W = net.blobs[net.inputs[0]].data.shape
	F = net.blobs[net.outputs[0]].data.shape[1]
	Nf = len(filenames)
	Hi, Wi, _ = imread(filenames[0]).shape
	allftrs = np.zeros((Nf, F))
	for i in range(0, Nf, N):
		in_data = np.zeros((N, C, H, W), dtype=np.float32)

		batch_range = range(i, min(i + N, Nf))
		batch_filenames = [filenames[j] for j in batch_range]
		Nb = len(batch_range)

		batch_images = np.zeros((Nb, 3, H, W))
		for j, fname in enumerate(batch_filenames):
			im = imread(fname)
			if len(im.shape) == 2:
				im = np.tile(im[:, :, np.newaxis], (1, 1, 3))
			# RGB -> BGR
			im = im[:, :, (2, 1, 0)]
			# mean subtraction [https://gist.github.com/ksimonyan/211839e770f7b538e2d8]
			im = im - np.array([103.939, 116.779, 123.68])
			# resize
			im = imresize(im, (H, W), 'bicubic')
			# get channel in correct dimension
			im = np.transpose(im, (2, 0, 1))
			batch_images[j, :, :, :] = im

		# insert into correct place
		in_data[0:len(batch_range), :, :, :] = batch_images

		# predict features
		ftrs = predict(net, in_data)

		for j in range(len(batch_range)):
			allftrs[i + j, :] = ftrs[j, :]

		print('Done ' + str(i + len(batch_range)) + '/' + str(len(filenames)) + ' files')

	return allftrs


def predict(net, in_data):
	"""
	Get the features for a batch of data using network

	Inputs:
	in_data: data batch
	"""

	out = net.forward(**{ net.inputs[0]: in_data })
	features = out[net.outputs[0]]

	return features


def makeSubmission(predictions, filenames, info='submission_finetuned_', discrete=False, threshold=0.1):
	print('Preparing submission file...')

	if discrete:
		argmax = np.argmax(predictions, axis=1)
		for i in range(0, len(predictions)):
			if predictions[i][argmax[1]] < threshold:
				continue
			for j in range(0, len(predictions[i])):
				predictions[i][j] = 0
			predictions[i][argmax[1]] = 1

	result = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3',
												'c4', 'c5', 'c6', 'c7',
												'c8', 'c9'])
	filenames = [f.split('/')[-1] for f in filenames]
	result.loc[:, 'img'] = pd.Series(filenames, index=result.index)
	timestr = time.strftime("%Y%m%d-%H%M%S")
	name = info
	if discrete:
		name += 'd_'
	result.to_csv("%s%s.csv" % (name, timestr), index=False)


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('-md', '--model_def', type=str, default='VGG_16_deploy_test.prototxt',
						help='Path to model definition prototxt')
	parser.add_argument('-m', '--model', type=str, default='VGG_16.caffemodel',
						help='Path to caffe model')
	parser.add_argument('-fd', '--filesdir', type=str, default='imgs/test',
						help='Path to a dir containing images')
	parser.add_argument('-g', '--gpu', action='store_true',
						help='Whether to use gpu training')

	args = parser.parse_args()
	params = vars(args)

	if args.gpu:
		caffe.set_mode_gpu()
	else:
		caffe.set_mode_cpu()

	net = caffe.Net(args.model_def, args.model, caffe.TEST)

	main(net, params['filesdir'])
