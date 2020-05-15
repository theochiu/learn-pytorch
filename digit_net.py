
import torch 
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import sys

RESUME = False

# PATH = "./digitnet.pth"
PATH = "./fashion-model-cnn3.pth"
# PATH = "./fashion-model.pth"

transform = transforms.Compose(
	[transforms.ToTensor()])

trainset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True,
	transform=transform)
testset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True,
	transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

classes = ["t-shirt/top", "pants", "long sleeve shirt", "dress", "coat",
			"sandal", "collared shirt", "sneaker", "bag", "ankle boot"]

def imshow(image):

	nping = image.numpy()
	plt.imshow(nping[0], cmap="gray")
	plt.show()
 
def image_main():

	dataiter = iter(trainloader)
	images, labels = dataiter.next()
	# lets cap it to 4...
	images = images[:3]
	labels = labels[:3]

	print(', '.join('%5s' % classes[label.item()] for label in labels))
	# print(labels)

	imshow(torchvision.utils.make_grid(images))

class DigitNet(nn.Module):
	def __init__(self):
		super(DigitNet, self).__init__()

		self.network = nn.Sequential(
			nn.Linear(784, 256),
			nn.SELU(),
			nn.Linear(256, 256),
			nn.SELU(),
			nn.Linear(256, 128),
			nn.SELU(),
			nn.Linear(128, 64),
			nn.SELU(),
			nn.Linear(64, 10),
		)

	def forward(self, x):
		x = x.view(x.size(0), -1)

		x = self.network(x)
		return x


class DigitCNN(nn.Module):
	def __init__(self):
		super(DigitCNN, self).__init__()

		self.convlayers = nn.Sequential(
			nn.Conv2d(1, 64, 5, padding=2),		# in_chan, out_chan, kernel
			nn.SELU(),
			nn.Conv2d(64, 64, 5, padding=2),
			nn.SELU(),
			# 28 x 28 x 64
			nn.MaxPool2d(2, 2),
			# 14 x 14 x 64

			nn.Conv2d(64, 128, 5, padding=2),
			nn.SELU(),
			# 14 x 14 x 128
			nn.Conv2d(128, 128, 5, padding=2),
			nn.SELU(),
			nn.MaxPool2d(2, 2),
			# 7 x 7 x 128

			nn.Conv2d(128, 256, 3, padding=1),
			nn.SELU(),
			# 7 x 7 x 256

			nn.Conv2d(256, 256, 3, padding=1),
			nn.SELU(),
			# 7 x 7 x 256

		)

		self.gap = nn.Sequential(
			nn.AdaptiveAvgPool2d((1, 128)),
			nn.Linear(128, 10)
		)

		self.denselayers = nn.Sequential(
			nn.Linear(256*7*7, 512),
			nn.SELU(),
			nn.Linear(512, 256),
			nn.SELU(),
			nn.Linear(256, 10),
		)

	def forward(self, x):
		x = self.convlayers(x)
		# x = self.gap(x)

		x = x.view(x.size(0), -1)	# flatten
		x = self.denselayers(x)

		return x

# net = DigitNet()
net = DigitCNN()

def train():
	lr = 0.01
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.95, nesterov=True)
	# optimizer = optim.Adagrad(net.parameters())
	# optimizer = optim.Adam(net.parameters(), lr=0.01)


	# load model if resume
	old_epoch = 0
	old_loss = float("inf")

	if RESUME:
		checkpoint = torch.load(PATH)
		net.load_state_dict(checkpoint["model_state_dict"])
		optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		old_epoch = checkpoint["epoch"] + 1
		lr = checkpoint["lr"]
		# optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.95, nesterov=True)
		old_loss = checkpoint['old_loss']
		print("loaded")


	for epoch in range(old_epoch, 15):
	# for epoch in range(old_epoch, old_epoch+15):
		running_loss = 0.0

		print("lr: {}".format(lr))

		for i, data in enumerate(trainloader, 0):
			inputs, labels = data
			optimizer.zero_grad()
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()

			if i%100==99:
				print('[%d, %5d] loss: %.5f' %
						(epoch + 1, i + 1, running_loss / 2000))

				if old_loss < running_loss and lr > 0.000001:
					lr /= 5
					print("lr: {}".format(lr))
					optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.95, nesterov=True)

				old_loss = running_loss
				running_loss = 0.0

		# save model per epoch
		torch.save({
			'epoch': epoch,
			'model_state_dict': net.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'old_loss': old_loss,
			'lr': lr,
		}, PATH)
		print("Finished epoch, model saved\n")


def test_model():
	checkpoint = torch.load(PATH)
	net.load_state_dict(checkpoint["model_state_dict"])

	correct = 0
	total = 0

	with torch.no_grad():
		for data in testloader:
			images, labels = data
			outputs = net(images)
			_, predicted = torch.max(outputs, 1)
			total += labels.size(0)
			correct += (predicted==labels).sum().item()
			c = (predicted == labels).squeeze()
	
	print("\nAccuracy: {}%".format(100*correct/total))
	print("Error: {:.5f}%".format(100 * (1 - correct / total)))


if __name__ == '__main__':


	if len(sys.argv) > 1 and sys.argv[1] == "-r":
		print("RESUME TRAIN MODE\n")
		RESUME = True
		train()
		test_model()

	elif len(sys.argv) > 1 and sys.argv[1] == "--test":
		print("TEST MODE\n")
		test_model()

	else:
		print("TRAIN MODE\n")
		train()
		test_model()

	# image_main()
	# train()
	# test_model()
	# print(net)
	# x,_ = trainset[0]
	# plt.imshow(x.numpy()[0], cmap="gray")
	# plt.show()

