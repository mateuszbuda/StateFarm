__author__ = 'mateusz'

import cPickle as pickle
import numpy as np
from sklearn.neural_network import MLPClassifier


featuresDir = "features/"
featuresPrefix = "features_c"
featuresExt = ".p"



def main():

	print "Loading features ..."
	features, labels = loadFeatures([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

	trainX, trainY, testX, testY = splitTrainTest(features, labels, train=0.5)


	print "Training MLP ..."
	mlp = trainMLP(trainX, trainY, hidden_layer_sizes=256)
	print mlp
	print "train error = " + str(mlp.score(trainX, trainY))

	print "Evaluating ..."
	score = mlp.score(testX, testY)
	print "test error = " + str(score)



def loadFeatures(classes):

	fFile = featuresDir + featuresPrefix + str(classes[0]) + featuresExt
	features = np.array(pickle.load( open( fFile, "rb" ) ))
	labels = np.ones([len(features), 1]) * classes[0]

	for c in classes[1:]:
		fFile = featuresDir + featuresPrefix + str(c) + featuresExt

		featuresC = np.array(pickle.load( open( fFile, "rb" ) ))
		features = np.vstack([features, featuresC])

		labelsC = np.ones([len(featuresC), 1]) * c
		labels = np.vstack([labels, labelsC])

	return features, labels.flatten()



def trainMLP(features, labels, activation='tanh', algorithm='adam', hidden_layer_sizes=(2048), alpha=0.0001):

	mlp = MLPClassifier(activation=activation, algorithm=algorithm, hidden_layer_sizes=hidden_layer_sizes, alpha=alpha)

	mlp.fit(features, labels)

	return mlp



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
