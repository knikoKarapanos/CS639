from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import numpy as np
import cv2

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
            classes.append(path.name)
            for image_path in path.iterdir():
                if np.random.random() < 0.1:
                    test_image_paths.append(image_path)
                else:
                    train_image_paths.append(image_path)
                
    train_image_paths, valid_image_paths = train_image_paths[:int(0.8*len(train_image_paths))], train_image_paths[int(0.8*len(train_image_paths)):]

    print("Train size {}, valid size {}, test size {}".format(len(train_image_paths), len(valid_image_paths), len(test_image_paths)))
    print("There are {} classes".format(len(classes)))

    class_data = ProduceClasses(classes)
    return class_data



def get_train_loader():
    train_dataset = ProduceDataset(train_image_paths, transform=data_transforms['train'])
    return DataLoader(train_dataset, batch_size=32, shuffle=True)

def get_valid_loader():
    valid_dataset = ProduceDataset(valid_image_paths, transform=data_transforms['valid'])
    return DataLoader(valid_dataset, batch_size=32, shuffle=True)

def get_test_loader():
    test_dataset = ProduceDataset(test_image_paths, transform=data_transforms['valid'])
    return DataLoader(test_dataset, batch_size=32, shuffle=True)