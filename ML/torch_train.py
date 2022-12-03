import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

from classes import *
from Data import *

def main():

    EPOCHS = 5
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
            # Save first 4 images as figure
            if epoch == 0 and i == 0:
                plt.ioff()
                plt.subplots_adjust(hspace=1, wspace=1.2)
                for j in range(4):
                    image = images[j]
                    image = image.numpy().transpose((1, 2, 0))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (50, 50))
                    image = np.clip(image, 0, 1)
                    plt.subplot(2, 2, j + 1)
                    plt.imshow(image)
                    plt.title(classes[labels[j]])
                plt.savefig("images/train_images.png")

            

            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 50 == 49:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    outfile = "model.pth" if not BINARY_FRESH_STALE else "model_binary.pth"
    torch.save(net.state_dict(), outfile)

if __name__ == "__main__":
    main()