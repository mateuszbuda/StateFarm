__author__ = 'mateusz'

import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
import pandas as pd
import time


featuresDir = 'features/'
featuresPrefix = 'features_c'
featuresExt = '.p'
featuresTest = 'features.p'
filenamesTest = 'filenames.p'
threshold = 0.75



def main():

	trainX, trainY = loadTrainSet(range(0, 10))
	testX, filenames = loadTestSet()

	for layerSize in [256, 512, 1024, 2048]:
		mlp = trainMLP(trainX, trainY, hidden_layer_sizes=layerSize)
		print('Train accuracy = ' + str(mlp.score(trainX, trainY)))

		print('Evaluating on test set...')
		predictions = mlp.predict_proba(testX)

		info = 'submission_' + str(layerSize) + '_'
		makeSubmission(predictions, filenames, info=info)
		makeSubmission(predictions, filenames, info=info, discrete=True)

	print('Done.')


def loadTrainSet(classes):

	print('Loading train set features...')

	fFile = featuresDir + featuresPrefix + str(classes[0]) + featuresExt
	features = np.array(pickle.load( open( fFile, "rb" ), encoding='latin1' ))
	labels = np.ones([len(features), 1]) * classes[0]

	for c in classes[1:]:
		fFile = featuresDir + featuresPrefix + str(c) + featuresExt

		featuresC = np.array(pickle.load( open( fFile, "rb" ), encoding='latin1' ))
		features = np.vstack([features, featuresC])

		labelsC = np.ones([len(featuresC), 1]) * c
		labels = np.vstack([labels, labelsC])

	return features, labels.flatten()



def loadTestSet():

	print('Loading test set features...')

	fFile = featuresDir + featuresTest
	features = np.array(pickle.load( open( fFile, "rb" ), encoding='latin1' ))
	fFile = featuresDir + filenamesTest
	filenames = np.array(pickle.load( open( fFile, "rb" ), encoding='latin1' ))

	return features, filenames



def trainMLP(features, labels, activation='tanh', algorithm='adam', hidden_layer_sizes=(2048), alpha=0.0001):

	print('Learning...')

	mlp = MLPClassifier(activation=activation, algorithm=algorithm, hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, verbose=True)
	print(mlp)

	mlp.fit(features, labels)

	return mlp



def makeSubmission(predictions, filenames, discrete=False, info='submission_'):

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
	testInd = indices[split+1:]

	trainX = X[trainInd]
	trainY = Y[trainInd]
	testX = X[testInd]
	testY = Y[testInd]

	return trainX, trainY, testX, testY



if __name__ == "__main__":

	main()
