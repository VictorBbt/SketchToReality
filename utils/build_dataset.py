# USAGE
# python build_dataset.py

# import necessary packages
import base_config
import numpy as np
import shutil
import os
import glob
import sys

sys.path.insert(0, os.path.abspath('../'))

def get_labels(sketchPath):
	return os.listdir(sketchPath)

def get_imgsName(labelPath):
	return os.listdir(labelPath)

def get_imgsPaths(dataPath):
	path = os.path.abspath(dataPath)
	path += '/*/*'
	print(path)
	list_imgPaths = []
	for file in glob.glob(path, recursive=True):
		list_imgPaths.append(file)
	print(len(list_imgPaths))
	return list_imgPaths
	
def copy_images(imagePaths, folder):
	# check if the destination folder exists and if not create it
	if not os.path.exists(folder):
		os.makedirs(folder)
	# loop over the image paths
	for path in imagePaths:
		# grab image name and its label from the path and create
		# a placeholder corresponding to the separate label folder
		imageName = path.split(os.path.sep)[-1]
		label = path.split(os.path.sep)[-2]
		labelFolder = os.path.join(folder, label)
		# check to see if the label folder exists and if not create it
		if not os.path.exists(labelFolder):
			os.makedirs(labelFolder)
		# construct the destination image path and copy the current
		# image to it
		destination = os.path.join(labelFolder, imageName)
		shutil.copy(path, destination)
		
def sketch_dataset():
    # load all the image paths and randomly shuffle them
    print("[INFO] loading image paths...")
    imagePaths = get_imgsPaths(base_config.SKETCH_DATASET_PATH)
    np.random.shuffle(imagePaths)
    # generate training and validation paths
    valPathsLen = int(len(imagePaths) * base_config.VAL_SPLIT)
    traintestPathsLen = len(imagePaths) - valPathsLen
    valPaths = imagePaths[traintestPathsLen:]
    testPathsLen = int(traintestPathsLen*base_config.TEST_SPLIT)
    trainPathsLen = traintestPathsLen - testPathsLen
    testPaths = imagePaths[trainPathsLen:]
    trainPaths = imagePaths[:trainPathsLen]
    # copy the training and validation images to their respective directories
    print("[INFO] copying training, test and validation images...")
    print(f'Train images: len {trainPathsLen} in {base_config.TRAIN}')
    copy_images(trainPaths, base_config.TRAIN)
    print(f'Validation  images: len {valPathsLen} in {base_config.VAL}')
    copy_images(valPaths, base_config.VAL)
    print(f'Test images: len {testPathsLen} in {base_config.TEST}')
    copy_images(testPaths, base_config.TEST)
    return None
