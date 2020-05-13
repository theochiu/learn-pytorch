
import torch 
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

PATH = "./digitnet.pth"

transform = transforms.Compose(
	[transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True,
	transform=transform)
testset = torchvision.datasets.MNIST(root="./data", train=False, download=True,
	transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

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

	print(' '.join('%5s' % label.item() for label in labels))
	# print(labels)

	imshow(torchvision.utils.make_grid(images))

class DigitNet(nn.Module):
	def __init__(self):
		super(DigitNet, self).__init__()

		self.network = nn.Sequential(
			nn.Linear(784, 128),
			nn.SELU(),
			nn.Linear(128, 128),
			nn.SELU(),
			nn.Linear(128, 64),
			nn.SELU(),
			nn.Linear(64, 10),
		)

	def forward(self, x):
		x = x.view(x.size(0), -1)

		x = self.network(x)
		return x

net = DigitNet()

def train():
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.95, nesterov=True)
	# optimizer = optim.Adam(net.parameters(), lr=0.01)

	for epoch in range(10):
		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data
			optimizer.zero_grad()
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()

			if i%100==99:
				print('[%d, %5d] loss: %.3f' %
						(epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0

	print("training finished")
	torch.save(net, PATH)
	print("model saved")

def test_model():
	net = torch.load(PATH)

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
	
	print("Accuracy: {}".format(100*correct/total))


if __name__ == '__main__':
	# image_main()
	train()
	test_model()
	# x,_ = trainset[0]
	# plt.imshow(x.numpy()[0], cmap="gray")
	# plt.show()

