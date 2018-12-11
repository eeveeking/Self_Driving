from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from glob import glob
import cv2


DEBUG = 1
VAL_DATA_SIZE = 600
TRAIN_DATA_SIZE = 2000
TEST_DATA_SIZE = 600


def preprocess(img):
	"Preprocess."
	img = cv2.resize(img,(IMGSIZE,IMGSIZE)).astype(np.float32)
	img -= [103.939, 116.779, 123.68]
	img = img / 255

	return img


def data_load():
	"Load and preprocess data."
	# data_dir = 'deploy/'
	TRAIN_FILE = glob('deploy/trainval/*/*_image.jpg')
	TEST_FILE = glob('deploy/test/*/*_image.jpg')
	TRAIN = 'train'
	VAL = 'val'
	TEST = 'test'

	# VGG-16 Takes 224x224 images as input, so we resize all of them
	data_transforms = {
	    TRAIN: transforms.Compose([
	        # Data augmentation is a good practice for the train set
	        # Here, we randomly crop the image to 224x224 and
	        # randomly flip it horizontally.
	        transforms.Resize(256),
			transforms.RandomResizedCrop(224),
	        transforms.ToTensor(),
	    ]),
	    VAL: transforms.Compose([
	        transforms.Resize(256),
	        transforms.CenterCrop(224),
	        transforms.ToTensor(),
	    ]),
	    TEST: transforms.Compose([
	        transforms.Resize(256),
	        transforms.CenterCrop(224),
	        transforms.ToTensor(),
	    ])
	}

	if DEBUG:
		VAL_FILE = TRAIN_FILE[5000:5000+VAL_DATA_SIZE]
		TRAIN_FILE = TRAIN_FILE[:TRAIN_DATA_SIZE]
		TEST_FILE = TEST_FILE[1000:1000+TEST_DATA_SIZE]

	image_datasets = {}
	image_datasets[TRAIN] = [preprocess(cv2.imread(fname)) for fname in TRAIN_FILE]
	image_datasets[VAL] = [preprocess(cv2.imread(fname)) for fname in VAL_FILE]
	image_datasets[TEST] = [preprocess(cv2.imread(fname)) for fname in TEST_FILE]


	# image_datasets = {
	#     x: datasets.ImageFolder(
	#         os.path.join(data_dir, x),
	#         transform=data_transforms[x]
	#     )
	#     for x in [TRAIN, VAL, TEST]
	# }

	dataloaders = {
	    x: torch.utils.data.DataLoader(
	        image_datasets[x], batch_size=8,
	        shuffle=True, num_workers=4
	    )
	    for x in [TRAIN, VAL, TEST]
	}

	dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL, TEST]}

	for x in [TRAIN, VAL, TEST]:
	    print("Loaded {} images under {}".format(dataset_sizes[x], x))

	print("Classes: ")
	class_names = image_datasets[TRAIN].classes
	print(image_datasets[TRAIN].classes)


def eval_model(vgg, criterion):
	"This helper function will give us the accuracy of our model on the test set."
    since = time.time()
    avg_loss = 0
    avg_acc = 0
    loss_test = 0
    acc_test = 0

    test_batches = len(dataloaders[TEST])
    print("Evaluating model")
    print('-' * 10)

    for i, data in enumerate(dataloaders[TEST]):
        if i % 100 == 0:
            print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

        vgg.train(False)
        vgg.eval()
        inputs, labels = data

        if use_gpu:
            inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
        else:
            inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)

        outputs = vgg(inputs)

        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        loss_test += loss.data[0]
        acc_test += torch.sum(preds == labels.data)

        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()

    avg_loss = loss_test / dataset_sizes[TEST]
    avg_acc = acc_test / dataset_sizes[TEST]

    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg loss (test): {:.4f}".format(avg_loss))
    print("Avg acc (test): {:.4f}".format(avg_acc))
    print('-' * 10)


def load_vgg16():
	# Load the pretrained model from pytorch
	vgg16 = models.vgg16_bn()
	vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
	print(vgg16.classifier[6].out_features) # 1000


	# Freeze training for all layers
	for param in vgg16.features.parameters():
	    param.require_grad = False

	# Newly created modules have require_grad=True by default
	num_features = vgg16.classifier[6].in_features
	features = list(vgg16.classifier.children())[:-1] # Remove last layer
	features.extend([nn.Linear(num_features, len(class_names))]) # Add our layer with 4 outputs
	vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
	print(vgg16)


def train_model(vgg, criterion, optimizer, scheduler, num_epochs=10):
	"Train model."
    since = time.time()
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_acc = 0.0

    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0

    train_batches = len(dataloaders[TRAIN])
    val_batches = len(dataloaders[VAL])

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)

        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0

        vgg.train(True)

        for i, data in enumerate(dataloaders[TRAIN]):
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)

            # Use half training dataset
            if i >= train_batches / 2:
                break

            inputs, labels = data

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = vgg(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_train += loss.data[0]
            acc_train += torch.sum(preds == labels.data)

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        print()
        # * 2 as we only used half of the dataset
        avg_loss = loss_train * 2 / dataset_sizes[TRAIN]
        avg_acc = acc_train * 2 / dataset_sizes[TRAIN]

        vgg.train(False)
        vgg.eval()

        for i, data in enumerate(dataloaders[VAL]):
            if i % 100 == 0:
                print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)

            inputs, labels = data

            if use_gpu:
                inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
            else:
                inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)

            optimizer.zero_grad()

            outputs = vgg(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss_val += loss.data[0]
            acc_val += torch.sum(preds == labels.data)

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        avg_loss_val = loss_val / dataset_sizes[VAL]
        avg_acc_val = acc_val / dataset_sizes[VAL]

        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg acc (val): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        print()

        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(vgg.state_dict())

    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))

    vgg.load_state_dict(best_model_wts)
    return vgg


def main():
	plt.ion()

	use_gpu = torch.cuda.is_available()
	if use_gpu:
	    print("Using CUDA")

	data_load()

	vgg16 = load_vgg16()

	# If you want to train the model for more than 2 epochs, set this to True after the first run
	resume_training = False

	if resume_training:
	    print("Loading pretrained model..")
	    vgg16.load_state_dict(torch.load('../input/vgg16-transfer-learning-pytorch/VGG16_v2-OCT_Retina.pt'))
	    print("Loaded!")

	if use_gpu:
	    vgg16.cuda() #.cuda() will move everything to the GPU side

	criterion = nn.CrossEntropyLoss()

	optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

	vgg16 = train_model(vgg16, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=2)
	torch.save(vgg16.state_dict(), 'VGG16_v2-OCT_Retina_half_dataset.pt')



if __name__ == '__main__':
	main()