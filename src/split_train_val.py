import pandas as pd
from shutil import move
import os
from random import shuffle

imgsPath = 'imgs/'

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


def splitDrivers(drivers, trainSize=0.75):
	subjectList = drivers['subject'].drop_duplicates().tolist()
	print(subjectList)
	print(len(subjectList))
	trainCount = int(len(subjectList) * trainSize)
	print(trainCount)
	shuffle(subjectList)
	trainDrivers = subjectList[:trainCount]
	validateDrivers = subjectList[trainCount+1:]
	print(validateDrivers)

	return trainDrivers, validateDrivers


def moveFiles(drivers, setList, setType):
	for line in drivers[drivers['subject'].isin(setList)].values:
		if not os.path.exists(imgsPath + setType + '/' + line[1]):
			os.makedirs(imgsPath + setType + '/' + line[1])
		move(imgsPath + 'train/' + line[1] + '/' + line[2], imgsPath + setType + '/' + line[1] + '/' + line[2])


if __name__ == "__main__":

	main()
