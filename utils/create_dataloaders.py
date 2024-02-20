import sys
import os
sys.path.insert(0, os.path.abspath('../'))
import base_config, build_dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch

# initialize our data augmentation functions
resize = transforms.Resize(size=(base_config.INPUT_HEIGHT,
        base_config.INPUT_WIDTH))
hFlip = transforms.RandomHorizontalFlip(p=0.25)
vFlip = transforms.RandomVerticalFlip(p=0.25)
rotate = transforms.RandomRotation(degrees=15)

# initialize our training and validation set data augmentation
# pipeline
trainTransforms = transforms.Compose([resize, hFlip, vFlip, rotate,
        transforms.ToTensor()])
valTransforms = transforms.Compose([resize, transforms.ToTensor()])
def visualize_batch(batch, classes, dataset_type):
	# initialize a figure
	batch_size = base_config.BATCH_SIZE
	n_rows = max(1, batch_size // 4)
	fig, axes = plt.subplots(n_rows, 4, figsize=(batch_size, batch_size))
	fig.suptitle(f"{dataset_type} batch")
	axes = axes.flatten()
	# loop over the batch size
	for i in range(0, base_config.BATCH_SIZE):
		# create a subplot
		ax = axes[i]
		# grab the image, convert it from channels first ordering to
		# channels last ordering, and scale the raw pixel intensities
		# to the range [0, 255]
		image = batch[0][i].cpu().numpy()
		image = image.transpose((1, 2, 0))
		image = (image * 255.0).astype("uint8")
		# grab the label id and get the label from the classes list
		idx = batch[1][i]
		label = classes[idx] # In the DataLoader class
		# show the image along with the label
		ax.imshow(image)
		ax.set_title(label)
		ax.axis("off")
	# show the plot
	plt.tight_layout()
	plt.show()
	
def split_dataset(show_batch = True):
    # We formatted our data directory to use ImageFolder from torchvision

    # initialize the training and validation dataset
    print("[INFO] loading the training and validation dataset...")
    trainDataset = ImageFolder(root=base_config.TRAIN,
            transform=trainTransforms)
    testDataset = ImageFolder(root=base_config.TEST, 
            transform=trainTransforms)
    valDataset = ImageFolder(root=base_config.VAL, 
            transform=valTransforms)
    print("[INFO] training dataset contains {} samples...".format(
            len(trainDataset)))
    print("[INFO] test dataset contains {} samples...".format(
            len(testDataset)))
    print("[INFO] validation dataset contains {} samples...".format(
            len(valDataset)))
	
    # create training and validation set dataloaders
    print("[INFO] creating training and validation set dataloaders...")
    trainDataLoader = DataLoader(trainDataset, 
            batch_size=base_config.BATCH_SIZE, shuffle=True)
    testDataLoader = DataLoader(testDataset, 
            batch_size=base_config.BATCH_SIZE, shuffle=True)
    valDataLoader = DataLoader(valDataset, batch_size=base_config.BATCH_SIZE)
	
    if show_batch:
        # grab a batch from both training and validation dataloader
        trainBatch = next(iter(trainDataLoader))
        testBatch = next(iter(testDataLoader))
        valBatch = next(iter(valDataLoader))
        # visualize the training and validation set batches
        print("[INFO] visualizing training and validation batch...")
        visualize_batch(trainBatch, trainDataset.classes, "train")
        visualize_batch(testBatch, testDataset.classes, "test")
        visualize_batch(valBatch, valDataset.classes, "val")

    return trainDataLoader, testDataLoader, valDataLoader