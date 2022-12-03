from collections import OrderedDict
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from Data import *
from classes import BINARY_FRESH_STALE


class_data = load_data()

test_loader = get_test_loader()

dataiter = iter(test_loader)
images, labels = next(dataiter)

net = Net()

if BINARY_FRESH_STALE:
    net.load_state_dict(torch.load("model_binary.pth"))
else:
    net.load_state_dict(torch.load("model.pth"))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

correct = {class_: 0 for class_ in classes}
total = {class_: 0 for class_ in classes}
wrongs = {class_: [] for class_ in classes}

all_labels = []
all_predicted = []

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        for label, preds in zip(labels, predicted):
            all_labels.append(label)
            all_predicted.append(preds)
            if label == preds:
                correct[class_data.idx_to_class[int(label)]] += 1
            else:
                # Add one to the correct class to run a leaderboard of most misclassified
                wrongs[class_data.idx_to_class[int(label)]].append(class_data.idx_to_class[int(preds)])
            total[class_data.idx_to_class[int(label)]] += 1


# Create confusion matrix
confusion_matrix = metrics.confusion_matrix(all_labels, all_predicted)
plt.figure(figsize=(10, 10))
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Reds)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45, ha="right")
plt.yticks(tick_marks, classes)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig("images/confusion_matrix.png")


# Print and plot accuracy
for classname, correct_count in correct.items():
    accuracy = 100 * float(correct_count) / total[classname]
    print("Accuracy of %5s : %2d %%" % (classname, accuracy))

# Plot Accuracy of each class
accs = [(100 * float(v) / total[c]) for c, v in correct.items()]
bar_colors = ["green" if accuracy > 90 else "red" for accuracy in accs]
plt.ioff()
plt.figure()
plt.bar(range(len(correct)), accs, align='center', color=bar_colors)
plt.xticks(range(len(correct)), list(correct.keys()), rotation=45, ha='right')
plt.yticks(range(0, 100, 10))
plt.ylabel("Accuracy (%)")
plt.xlabel("Class")
plt.title("Accuracy of each class")
plt.tight_layout()
plt.savefig("images/accuracy.png")

# Print overall accuracy
print("Overall accuracy: %2d %%" % (100 * sum(correct.values()) / sum(total.values())))


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
