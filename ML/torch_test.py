import torch

from Data import *


class_data = load_data()

test_loader = get_test_loader()

dataiter = iter(test_loader)
images, labels = next(dataiter)

net = Net()

net.load_state_dict(torch.load("model.pth"))

outputs = net(images)

_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % class_data.idx_to_class[int(predicted[j])] for j in range(4)))

correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy on the network of %d images is %.2f %%" % (total, 100 * correct / total))