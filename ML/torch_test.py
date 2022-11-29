from collections import OrderedDict
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

correct = {class_: 0 for class_ in classes}
total = {class_: 0 for class_ in classes}
wrongs = {class_: [] for class_ in classes}

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        for label, preds in zip(labels, predicted):
            if label == preds:
                correct[class_data.idx_to_class[int(label)]] += 1
            else:
                # Add one to the correct class to run a leaderboard of most misclassified
                wrongs[class_data.idx_to_class[int(label)]].append(class_data.idx_to_class[int(preds)])
            total[class_data.idx_to_class[int(label)]] += 1

for classname, correct_count in correct.items():
    accuracy = 100 * float(correct_count) / total[classname]
    print("Accuracy of %5s : %2d %%" % (classname, accuracy))

for classname, wrong_count in wrongs.items():
    """
    Print the top wrong predictions for each class
    """
    # Get total count of each class in list
    wrong_count = {i: wrong_count.count(i) for i in wrong_count}
    # Sort by count
    wrong_count = OrderedDict(sorted(wrong_count.items(), key=lambda x: x[1], reverse=True))
    # Print top 5
    if (len(wrong_count) > 0):
        print("Top 5 wrong predictions for %s:" % classname)
        for i, (class_, count) in enumerate(wrong_count.items()):
            if i == 5:
                break
            print("\t%s: %d" % (class_, count))