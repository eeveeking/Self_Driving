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
from random import randint


DEBUG = 0
VAL_DATA_SIZE = 30
TRAIN_DATA_SIZE = 100
TEST_DATA_SIZE = 30
IMGSIZE = 224
CROPSIZE = 224

TRAIN_FILE = glob('deploy/trainval/*/*_image.jpg')
TEST_FILE = glob('deploy/test/*/*_image.jpg')
TRAIN = 'train'
VAL = 'val'
TEST = 'test'


def preprocess(img):
	"Preprocess."
	img = cv2.resize(img,(IMGSIZE,IMGSIZE)).astype(np.float32)
	# randx = randint(0,32)
	# randy = randint(0,32)
	# crop_img = img[randx:randx+CROPSIZE, randy:randy+CROPSIZE]
	img -= [103.939, 116.779, 123.68]
	img = img / 255

	img = np.einsum('ijk->kji', img)
	# print(img.shape)
	return img


def data_load():
	TRAIN_FILE = glob('deploy/trainval/*/*_image.jpg')
	TEST_FILE = glob('deploy/test/*/*_image.jpg')

	"Load and preprocess data."
	# data_dir = 'deploy/'

	# Read labels
	with open('labels.csv', 'r') as f:
		lines = f.readlines()
	lines = [line.rstrip('\n') for line in lines]

	ALL_LABEL = []
	for line in lines[1:]:
		# TEMP_LABEL = [0,0,0]
		# TEMP_LABEL[int(line.split(',')[1])] = 1
		# TEMP_LABEL[randint(0,2)] = 1
		ALL_LABEL.append(int(line.split(',')[1]))
	ALL_LABEL = np.array(ALL_LABEL)

	print(len(TRAIN_FILE), len(ALL_LABEL))

	CUR_LABEL_LEN = len(ALL_LABEL)
	for idx in range(CUR_LABEL_LEN):
		if ALL_LABEL[idx] == 0:
			if randint(0,1) == 0:
				TRAIN_FILE.append(TRAIN_FILE[idx])
				ALL_LABEL = np.append(ALL_LABEL, ALL_LABEL[idx])
		elif ALL_LABEL[idx] == 2:
			if randint(0,3) == 0:
				TRAIN_FILE.append(TRAIN_FILE[idx])
				ALL_LABEL = np.append(ALL_LABEL, ALL_LABEL[idx])

	print(len(TRAIN_FILE), len(ALL_LABEL))
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
	else:
		VAL_FILE = TRAIN_FILE[:2]


	if DEBUG:
		TRAIN_LABEL = ALL_LABEL[:TRAIN_DATA_SIZE]
		VAL_LABEL = ALL_LABEL[5000:5000+VAL_DATA_SIZE]
		print(len(TRAIN_FILE))
	else:
		TRAIN_LABEL = ALL_LABEL
		VAL_LABEL = ALL_LABEL[:2]

	image_datasets = {}
	print("Preprocessing...")
	
	image_datasets[TRAIN] = [(preprocess(cv2.imread(TRAIN_FILE[idx])), TRAIN_LABEL[idx]) for idx in range(len(TRAIN_FILE))]
	image_datasets[VAL] = [(preprocess(cv2.imread(VAL_FILE[idx])), VAL_LABEL[idx]) for idx in range(len(VAL_FILE))]
	image_datasets[TEST] = [preprocess(cv2.imread(fname)) for fname in TEST_FILE]
	print("Preprocessing done...")


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
		for x in [TRAIN, VAL]
	}

	dataloaders[TEST] = torch.utils.data.DataLoader(image_datasets[TEST], batch_size=8, shuffle=False, num_workers=4)

	dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL, TEST]}

	for x in [TRAIN, VAL, TEST]:
		print("Loaded {} images under {}".format(dataset_sizes[x], x))

	# print("Classes: ")
	# class_names = image_datasets[TRAIN].classes
	# print(image_datasets[TRAIN].classes)
	return dataloaders, dataset_sizes


def eval_model(vgg, criterion, use_gpu, dataloaders):
	"This helper function will give us the accuracy of our model on the test set."
	since = time.time()
	avg_loss = 0
	avg_acc = 0
	loss_test = 0
	acc_test = 0

	test_batches = len(dataloaders[TEST])
	print("Evaluating model")
	print('-' * 10)

	res = []

	for i, data in enumerate(dataloaders[TEST]):
		if i % 100 == 0:
			print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

		vgg.train(False)
		vgg.eval()
		# inputs, labels = data
		inputs = data

		if use_gpu:
			# inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
			inputs = Variable(inputs.cuda(), volatile=True)
		else:
			# inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
			inputs = Variable(inputs, volatile=True)

		outputs = vgg(inputs)

		_, preds = torch.max(outputs.data, 1)
		
		res += preds.tolist()

		# loss = criterion(outputs, labels)

		# loss_test += loss.item()
		# acc_test += torch.sum(preds == labels.data)

		del inputs, outputs, preds
		torch.cuda.empty_cache()

	# avg_loss = loss_test / dataset_sizes[TEST]
	# avg_acc = acc_test / dataset_sizes[TEST]

	elapsed_time = time.time() - since
	print()
	# print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
	# print("Avg loss (test): {:.4f}".format(avg_loss))
	# print("Avg acc (test): {:.4f}".format(avg_acc))
	print('-' * 10)

	return res


def load_vgg16():
	# Load the pretrained model from pytorch
	vgg16 = models.vgg16_bn()
	vgg16.load_state_dict(torch.load("../vgg16_bn.pth"))
	print(vgg16.classifier[6].out_features) # 1000


	# Freeze training for all layers
	for param in vgg16.features.parameters():
		param.require_grad = False

	# Newly created modules have require_grad=True by default
	num_features = vgg16.classifier[6].in_features
	features = list(vgg16.classifier.children())[:-1] # Remove last layer
	features.extend([nn.Linear(num_features, 3)]) # Add our layer with 4 outputs
	vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
	print(vgg16)

	return vgg16


def train_model(vgg, criterion, optimizer, scheduler, dataloaders, use_gpu, dataset_sizes, num_epochs=10):
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
				if i > 0:
					print("Accuracy so far:", int(acc_train)/(i*8))

			# Use half training dataset
			# if i >= train_batches / 2:
			# 	break

			inputs, labels = data

			# print(type(inputs))
			# print(inputs.size())
			if use_gpu:
				inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
			else:
				inputs, labels = Variable(inputs), Variable(labels)

			optimizer.zero_grad()

			outputs = vgg(inputs)

			# print(outputs)

			_, preds = torch.max(outputs.data, 1)
			loss = criterion(outputs, labels)

			loss.backward()
			optimizer.step()

			loss_train += loss.item()
			# print(preds, labels.data, torch.sum(preds == labels.data))
			acc_train += torch.sum(preds == labels.data)
			print(preds, labels.data)

			del inputs, labels, outputs, preds
			torch.cuda.empty_cache()

		print()
		# * 2 as we only used half of the dataset
		avg_loss = loss_train / dataset_sizes[TRAIN]
		avg_acc = int(acc_train) / dataset_sizes[TRAIN]
		print(avg_acc)

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

			loss_val += loss.item()
			acc_val += torch.sum(preds == labels.data)

			del inputs, labels, outputs, preds
			torch.cuda.empty_cache()

		avg_loss_val = loss_val / dataset_sizes[VAL]
		avg_acc_val = int(acc_val) / dataset_sizes[VAL]

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

	dataloaders, dataset_sizes = data_load()

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

	vgg16 = train_model(vgg16, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, use_gpu, dataset_sizes, num_epochs=1)
	torch.save(vgg16.state_dict(), 'VGG16_v2-OCT_Retina_half_dataset.pt')

	res = eval_model(vgg16, criterion, use_gpu, dataloaders)
	print(res)

	label_output = 'guid/image,label' + '\n'
	for i in range(len(res)):
		label_output += str(TEST_FILE[i])
		label_output += ',' + str(res[i]) + '\n'

	label_output = label_output.replace('deploy/test/', '')
	label_output = label_output.replace('_image.jpg', '')

	with open('output.csv', 'w') as f:
		f.write(label_output)


if __name__ == '__main__':
	main()
