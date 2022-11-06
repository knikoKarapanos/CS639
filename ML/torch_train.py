import time
import json
import copy

import matplotlib.pyplot as plt
from collections import OrderedDict


import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn

from classes import *
from Data import *

def main():

    EPOCHS = 2
    load_data()
    train_loader = get_train_loader()

    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    # print(images.shape)
    # print(labels.shape)


    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    outfile = "model.pth"
    torch.save(net.state_dict(), outfile)

if __name__ == "__main__":
    main()