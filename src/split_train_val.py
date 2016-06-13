import pandas as pd
from shutil import move
import os
from random import shuffle
import  numpy as np

imgsPath = 'imgs/'

def main():
	driversList = readDriversList()

	driversList = driversList.reindex(np.random.permutation(driversList.index))

	driversList = driversList[['img', 'classname']]

	driversList['classname'] = driversList['classname'].str.replace("c", "").astype(int)

	msk = np.random.rand(len(driversList)) < 0.8

	train = driversList[msk]
	valid = driversList[~msk]

	train.to_csv('trainset.csv', index=False, header=False, sep=" ")
	valid.to_csv('validset.csv', index=False, header=False, sep=" ")


def readDriversList():
	return pd.read_csv('driver_imgs_list.csv', delimiter=",")


if __name__ == "__main__":

	main()
