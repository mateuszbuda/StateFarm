import pandas as pd
from shutil import move
import os


def main():
	driversList = readDriversList()
	trainSubjects, validateSubjects = splitDrivers(driversList)

	moveFiles(driversList, validateSubjects, 'validation')

	valImages = driversList[driversList['subject'].isin(validateSubjects)][['img', 'classname']]
	trainImages = driversList[driversList['subject'].isin(trainSubjects)][['img', 'classname']]

	trainImages['classname'] = trainImages['classname'].str.replace("c", "").astype(int)
	valImages['classname'] = valImages['classname'].str.replace("c", "").astype(int)

	trainImages.to_csv('trainset.csv', index=False, header=False, sep=" ")
	valImages.to_csv('validationset.csv', index=False, header=False, sep=" ")


def readDriversList():
	return pd.read_csv('driver_imgs_list.csv', delimiter=",")


def splitDrivers(drivers, trainSize=0.8):
	subjectList = drivers['subject'].drop_duplicates()
	trainCount = int(len(subjectList) * trainSize)
	trainDrivers = subjectList[:trainCount]
	validateDrivers = subjectList[trainCount + 1:]

	return trainDrivers.values, validateDrivers.values


def moveFiles(drivers, setList, setType):
	for line in drivers[drivers['subject'].isin(setList)].values:
		if not os.path.exists('imgs/' + setType + '/' + line[1]):
			os.makedirs('imgs/' + setType + '/' + line[1])
		move('imgs/train/' + line[1] + '/' + line[2], 'imgs/' + setType + '/' + line[1] + '/' + line[2])


if __name__ == "__main__":

	main()
