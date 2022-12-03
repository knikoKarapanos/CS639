from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

from classes import *

train_image_paths = []
test_image_paths = []
valid_image_paths = []
classes = []

class ProduceClasses:
    def __init__(self, classes):
        self.classes = classes
        self.idx_to_class = {}
        self.class_to_idx = {}
        self._build()
    
    def _build(self):
        for idx, class_ in enumerate(self.classes):
            self.idx_to_class[idx] = class_
            self.class_to_idx[class_] = idx
        

class ProduceDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.class_data = ProduceClasses(classes)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        try:
            image = cv2.resize(image, (50, 50))
        except:
            import pdb; pdb.set_trace()

        label = image_path.parent.name
        label = self.class_data.class_to_idx[label]
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

def load_data():
    global train_image_paths
    global test_image_paths
    global valid_image_paths
    for path in Path('../data').iterdir():
        if path.is_dir():
            if BINARY_FRESH_STALE:
                fresh_stale = path.name.split("_")[0]
                if fresh_stale not in classes:
                    classes.append(fresh_stale)
            else:
                classes.append(path.name)
            for image_path in path.iterdir():
                if np.random.random() < 0.05:
                    test_image_paths.append(image_path)
                else:
                    train_image_paths.append(image_path)
                

    print("Train size {}, test size {}".format(len(train_image_paths), len(test_image_paths)))
    print("There are {} classes".format(len(classes)))

    # Save 1 image from each class in 1 figure
    
    plt.ioff()
    print("One image from each class")
    for class_ in classes:
        path = Path("../data") / class_
        image_path = list(path.iterdir())[0]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (100, 100))
        # Create a 6x2 subplot
        plt.subplots_adjust(hspace=1, wspace=1.2)
        plt.subplot(3, 4, classes.index(class_) + 1)
        plt.imshow(image)
        plt.title(class_)
    plt.savefig("images/classes.png")



    class_data = ProduceClasses(classes)
    return class_data



def get_train_loader():
    train_dataset = ProduceDataset(train_image_paths, transform=data_transforms['train'])
    return DataLoader(train_dataset, batch_size=32, shuffle=True)

def get_test_loader():
    test_dataset = ProduceDataset(test_image_paths, transform=data_transforms['valid'])
    return DataLoader(test_dataset, batch_size=32, shuffle=True)