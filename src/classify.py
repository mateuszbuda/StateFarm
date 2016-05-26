__author__ = 'mateusz'

import pickle
import numpy as np
from sknn.mlp import Layer, Classifier
import pandas as pd
import time

trainFeaturesDir = 'trainFeatures/'
validationFeaturesDir = 'validationFeatures/'
testFeaturesDir = 'testFeatures/'
featuresPrefix = 'features_c'
featuresExt = '.p'


def main():
	trainX, trainY = loadTrainSet(trainFeaturesDir)
	validX, validY = loadTrainSet(validationFeaturesDir)
	testX, filenames = loadTestSet()

	for layerSize in [256, 512, 1024]:
		mlp = trainMLP(trainX, trainY, validX, validY, hidden_layer_size=layerSize)
		print('Train accuracy = ' + str(mlp.score(trainX, trainY)))
		print('Validation accuracy = ' + str(mlp.score(validX, validY)))

		print('Evaluating on test set...')
		predictions = mlp.predict_proba(testX)

		info = 'submission_' + str(layerSize) + '_'
		makeSubmission(predictions, filenames, info=info)
		makeSubmission(predictions, filenames, info=info, discrete=True)

	print('Done.')


def loadTrainSet(dir, classes=range(0, 10)):
	print('Loading train set features...')

	fFile = trainFeaturesDir + featuresPrefix + str(classes[0]) + featuresExt
	features = np.array(pickle.load(open(fFile, "rb"), encoding='latin1'))
	labels = np.ones([len(features), 1]) * classes[0]

	for c in classes[1:]:
		fFile = trainFeaturesDir + featuresPrefix + str(c) + featuresExt

		featuresC = np.array(pickle.load(open(fFile, "rb"), encoding='latin1'))
		features = np.vstack([features, featuresC])

		labelsC = np.ones([len(featuresC), 1]) * c
		labels = np.vstack([labels, labelsC])

	return features, labels.flatten()


def loadTestSet():
	print('Loading test set features...')

	fFile = testFeaturesDir + 'features.p'
	features = np.array(pickle.load(open(fFile, "rb"), encoding='latin1'))
	fFile = testFeaturesDir + 'filenames.p'
	filenames = np.array(pickle.load(open(fFile, "rb"), encoding='latin1'))

	return features, filenames


def trainMLP(trainX, trainY, validationX, validationY, activation='Tanh', algorithm='rmsprop',
			 hidden_layer_size=2048, alpha=0.0001):
	print('Learning...')

	trainX, trainY = shuffle(trainX, trainY)
	validationX, validationY = shuffle(validationX, validationY)

	mlp = Classifier(
		layers=[
			Layer(activation, units=hidden_layer_size),
			Layer("Softmax", units=len(np.unique(trainY)))
		], learning_rule=algorithm,
		learning_rate=0.01,
		learning_momentum=0.9,
		batch_size=256,
		n_stable=10,
		n_iter=200,
		regularize="L2",
		weight_decay=alpha,
		loss_type="mcc",
		valid_set=(validationX, validationY),
		verbose=True)

	print(mlp)

	mlp.fit(trainX, trainY)

	return mlp


def shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def makeSubmission(predictions, filenames, info='submission_', discrete=False, threshold=0.1):
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


def splitTrainTest(X, Y, train=0.7, shuffle=True):
	indices = range(len(X))

	if shuffle:
		np.random.shuffle(indices)

	split = int(round(len(indices) * train))
	trainInd = indices[:split]
	testInd = indices[split + 1:]

	trainX = X[trainInd]
	trainY = Y[trainInd]
	testX = X[testInd]
	testY = Y[testInd]

	return trainX, trainY, testX, testY


if __name__ == "__main__":

	# TODO: parameters: hidden layers, feature folders, threshold

	main()
