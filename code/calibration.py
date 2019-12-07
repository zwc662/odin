# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""

from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from scipy import misc
import calMetric as m
import calData as d
import math
import pickle
import matplotlib.pyplot as plt

start = time.time()
#loading data sets

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
])




# loading neural network

# Name of neural networks
# Densenet trained on CIFAR-10:		 densenet10
# Densenet trained on CIFAR-100:		densenet100
# Densenet trained on WideResNet-10:	wideresnet10
# Densenet trained on WideResNet-100:   wideresnet100
#nnName = "densenet10"

#imName = "Imagenet"



criterion = nn.CrossEntropyLoss()

def arg_parser():
	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
	parser.add_argument('--batch-siz', type=int, default=64, metavar='N',
						help='input batch size for training (default: 64)')
	parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
						help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=100, metavar='N',
						help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
						help='learning rate (default: 0.01)')
	parser.add_argument('--alpha', type=float, default=0.99, metavar='ALPHA',
						help='alpha (default: 0.01)')
	parser.add_argument('--decay', type=float, default=0.0001, metavar='DECAY',
						help='weight decay (default: 0.001)')
	parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
						help='SGD momentum (default: 0.5)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=10, metavar='N',
						help='how many batches to wait before logging training status')
	
	parser.add_argument('--save-model', action='store_true', default=True,
						help='For Saving the current Model')
	args = parser.parse_args()
	
	return args

def main(nnName, dataName, CUDA_DEVICE, epsilon):
	net1 = torch.load("../models/{}.pth".format(nnName), map_location = 'cuda')
	optim = torch.optim.SGD(net1.parameters(), lr = epsilon)

	if CUDA_DEVICE is not None:
		net1.cuda(CUDA_DEVICE)
		device = torch.device('cuda') 

	if nnName == "densenet10" or nnName == "wideresnet10": 
		testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
		testloaderIn = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)
		trainloaderIn = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
	if nnName == "densenet100" or nnName == "wideresnet100": 
		testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform)
		testloaderIn = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)
		trainloaderIn = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

	testsetout = torchvision.datasets.ImageFolder("../data/{}".format(dataName), transform=transform)
	testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=1, shuffle=False, num_workers=2)

	temperature = torch.ones([1], device=device)
	legend = '_'.join([nnName, dataName, 'temperature'])

	try:
		net1, temperature = pickle.load(open('../calibration/' + legend + '.pt', 'rb'))
		print("Loaded model and temperature")
	except:
		pass

	for epoch in range(100):
		print("Epoch {} begins".format(epoch))
		train(net1, device, trainloaderIn, epoch, optim)
		temperature = fit_temperature(net1, device, testloaderIn, temperature,  epoch, epsilon)
		pickle.dump((net1, temperature), open('../calibration/' + legend + '.pt', 'wb'))
		#model, temperature = pickle.load(open('./checkpoints/cifar/cifar_model_temperature.pt', 'rb'))
		test(net1, device, testloaderOut, epoch)
		temperature_scaling_test(net1, device, testloaderOut, temperature, legend)

def train(model, device, dataloader, epoch, optim):
	criterion = nn.CrossEntropyLoss()
	correct = 0
	tot = 0
	tot_loss = 0.
	tot_iter = 0.
	for i, data in enumerate(dataloader):
		images, labels = data
		images, labels = images.to(device), labels.to(device)
		optim.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optim.step()
		
		tot_loss += loss.item()
		_, predicted = torch.max(outputs, dim = 1)
		correct += (predicted == labels).sum().item()
		tot += images.size()[0]
	print("Epoch: %d   Avg loss: %.3f  Acc: %.3f" % (epoch, tot_loss/tot, float(correct)/tot)) 

def test(model, device, dataloader, epoch):
	correct = 0
	tot = 0
	for i, data in enumerate(dataloader):
		images, labels = data
		images, labels = images.to(device), labels.to(device)
		outputs = model(images)
		_, predicted = torch.max(outputs, dim = 1)
		correct += (predicted == labels).sum().item()
		tot += images.size()[0]
	print("Epoch: %d Acc: %.3f" % (epoch, float(correct)/tot))

		

def fit_temperature(model, device, trainloader, temperature_init, epoch, epsilon):
	criterion = nn.NLLLoss(reduction='sum')
	running_loss = 0.

	temperature = temperature_init
	correct = 0.
	tot = 0.
	for i, data in enumerate(trainloader, 0):
		temperature.requires_grad_()
		images, labels = data
		images = images.to(device)
		labels = labels.to(device)
		with torch.no_grad():
			logits = model(images).detach()
			_, predicted = torch.max(logits.data, 1)
			correct += (predicted == labels).sum().item()
			tot += images.size()[0]
			logits_nllloss = -criterion(logits, labels)
		
		logits_confs = torch.nn.functional.softmax(logits/(temperature * temperature), dim = 1)

		#logits_expectation = torch.bmm(logits.view(logits.size()[0], 1, -1), logits_confs.view(logits.size()[0], -1, 1)).flatten().sum()
		#logits_entropy = -(logits_confs * torch.log(logits_confs)).sum(dim = 1).flatten().sum()
		logits_confs_nllloss = criterion(torch.log(logits_confs), labels)
		loss = logits_confs_nllloss
		loss.backward()
		with torch.no_grad():
			if math.isnan(temperature.grad.item()):
				break
			temperature = temperature - epsilon * temperature.grad
			#temp.grad.zero_()
		
		# print statistics
		running_loss += loss.item()
		if i % 20 == 1:	# print every 200 mini-batches
			print('[%d, %5d] accuracy: %.3f, loss: %.3f' % (epoch, i, correct/tot, running_loss / 20))
			running_loss = 0.0
			correct = 0.
			tot = 0.
	print('Final temperature ', temperature)
	return temperature


def temperature_scaling_test(model, device, testloader, temperature, legend):
	xs = []
	ys = []
	targets = []

	zs = []
	for data, label in testloader:
		images = data.to(device)
		labels = label.to(device)
		with torch.no_grad():
			inputs = model(images)
			outputs = torch.nn.functional.log_softmax(inputs/(temperature * temperature), dim = 1)
		conf, predicted = torch.max(torch.exp(outputs).data, 1)
		xs = xs + conf.detach().cpu().numpy().tolist()	
		targets = targets + (predicted == labels).detach().cpu().numpy().tolist()

	x_pdfs = []
	y_pdfs = []
	den = 10
	x_intervs = np.linspace(np.min(xs), np.max(xs), den + 1)
	for i in range(den):
		x_pdf = x_intervs[i + 1]
		conf_interv_idx = ((np.asarray(xs) <= x_intervs[i + 1]) *  (np.asarray(xs) >= x_intervs[i])).astype(np.float) 
		pos_idx = (np.asarray(targets) >= 1).astype(np.float)
		num_conf_interv_pos = np.sum(conf_interv_idx * pos_idx)
		#print('number of pos in conf interv', num_conf_interv_pos)
		num_conf_interv_tot = np.sum(conf_interv_idx) 
		#print('number of tot in conf interv', num_conf_interv_tot)
		if num_conf_interv_tot == 0.0:
			y_pdf = 0.
		else:
			y_pdf = num_conf_interv_pos/num_conf_interv_tot
			x_pdfs.append(x_pdf)
			y_pdfs.append(y_pdf)
			print("{}~{}: {}/{}={}".format(x_intervs[i], x_intervs[i+1], num_conf_interv_pos, num_conf_interv_tot, y_pdf))
	plot_curv(x_pdfs, 'conf', y_pdfs, 'acc', legend + '_pdf', zs = None)	

def plot_curv(xs, x_label, ys, y_label, legend, zs = None):
	plt.xticks(np.linspace(-0.1, 1.0, num = 12))
	plt.yticks(np.linspace(-0.1, 1.0, num = 12))
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.plot(xs, ys, 'ro-')

	if zs is not None:
		bars = [None for i in range(len(zs[0]))]
		bars[0] = plt.bar(xs, [z[0]/np.sum(z) for z in zs], width = 0.05)
		sum_z = np.array([z[0]/np.sum(z) for z in zs])
		for i in range(1, len(zs[0])):
			bars[i] = plt.bar(xs, [z[i]/np.sum(z) for z in zs], width = 0.05, bottom = sum_z)
			sum_z += np.array([z[i]/np.sum(z) for z in zs])
		plt.legend(bars, [str(36 * (int(i%5) - 5 +  int(i/5) * 5)) for i in range(10)])
		
	plt.bar(xs, xs, width = 0.05)

	plt.savefig('../calibration/' + legend, format="png")
	
	plt.close()


if __name__ == "__main__":
	nnName = 'densenet10'
	dataName = 'Imagenet_resize'

	CUDA_DEVICE = 0
	epsilon = 1e-3
	main(nnName, dataName, CUDA_DEVICE, epsilon)
