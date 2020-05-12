import torch 
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

PATH = "./alex_net2.pth"


# turn the image into a tensor and then normalize it
transform = transforms.Compose(
	[transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# Load the training data
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True,
											 transform=transform)

# put the training data into a loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True,
											num_workers=2)

testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True,
										transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False,
											num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
		   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(image):
	image = image / 2 + 0.5     # unnormalize
	nping = image.numpy()
	plt.imshow(np.transpose(nping, (1, 2, 0)))
	plt.show()
	plt.savefig('figure.png') 

class AlexNet(nn.Module):

	def __init__(self, num_classes=10):
		super(AlexNet, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
			nn.ELU(),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(64, 192, kernel_size=3, padding=1),
			nn.ELU(),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.ELU(),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ELU(),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ELU(),
			nn.MaxPool2d(kernel_size=2),
		)
		self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 6 * 6, 4096),
			nn.ELU(),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ELU(),
			nn.Linear(4096, num_classes),
		)

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x

net = AlexNet()
# print(net)

def train():
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.01, nesterov=False)
	# optimizer = optim.Adagrad(net.parameters())

	for epoch in range(2):
		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data
			optimizer.zero_grad()
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()

			# if i%2000 == 1999:
			print("[%d, %5d] loss: %.3f" % (epoch+1, i+1, running_loss))
			running_loss = 0.0

	print("Training finished")
	torch.save(net, PATH)
	print("Model saved")


def test_model():
	# load model
	net = (torch.load(PATH))



	correct = 0
	total = 0
	class_correct = list(0. for i in range(10))
	class_total = list(0. for i in range(10))

	with torch.no_grad():
		for data in testloader:
			images, labels = data
			outputs = net(images)
			_, predicted = torch.max(outputs, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			c = (predicted == labels).squeeze()
			for i in range(4):
				label = labels[i]
				class_correct[label] += c[i].item()
				class_total[label] += 1


	print('Accuracy of the network on the 10000 test images: %d %%' % (
		100 * correct / total))

	for i in range(10):
		print('Accuracy of %5s : %2d %%' % (
			classes[i], 100 * class_correct[i] / class_total[i]))

if __name__ == '__main__':
	train()
	# test_model()

