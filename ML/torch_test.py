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

correct = {class_: 0 for class_ in classes}
total = {class_: 0 for class_ in classes}

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        for label, preds in zip(labels, predicted):
            if label == preds:
                correct[class_data.idx_to_class[int(label)]] += 1
            total[class_data.idx_to_class[int(label)]] += 1

for classname, correct_count in correct.items():
    accuracy = 100 * float(correct_count) / total[classname]
    print("Accuracy of %5s : %2d %%" % (classname, accuracy))