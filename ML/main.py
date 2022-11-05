import pickle
import numpy as np
import cv2
import logging

from pathlib import Path

DATADIR = Path(__file__).resolve().parents[1] / "data"
LABELS = DATADIR / "ImageLabels.txt"

TESTING_RATIO = 0.1 # How much of the data should be used for testing

training_imgs = []
testing_imgs = []

IMG_SIZE = 50

ALL_SAMPLES = -1 # Set sample size to ALL_SAMPLES to use all samples
SAMPLE_SIZE = 1000 # Set this lower if you want to test the code

def load_labels():
    with open(LABELS) as f:
        labels = f.read().splitlines()
    
    logging.debug("Loaded labels: %s", labels)
    return labels

def save_data(data: "list[list[int], int]", filename: str) -> np.array:
    X = [] # Images (input)
    y = [] # Label (output)
    for features, label in data:
        X.append(features)
        y.append(label)
    
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    data_outfile =open(filename + "_X.pickle", "wb")
    pickle.dump(X, data_outfile)
    data_outfile.close()
    label_outfile = open(filename + "_y.pickle", "wb")
    pickle.dump(y, label_outfile)
    label_outfile.close()
    logging.info("Saved %s", filename)

    

def load_data(labels):
    for label in labels:
        path = DATADIR / label
        for img in path.iterdir():
            if img.suffix not in [".jpg", ".png"]:
                continue
            logging.debug("Loading image %s", img)
            if np.random.random() < TESTING_RATIO:
                img_array = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                testing_imgs.append([new_array, labels.index(label)])
            else:
                img_array = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_imgs.append([new_array, labels.index(label)])
    
    logging.info("Loaded %d training images and %d testing images", len(training_imgs), len(testing_imgs))
    testing_data = save_data(testing_imgs, "testing_data")
    training_data = save_data(training_imgs, "training_data")




def main():
    logging.basicConfig(level=logging.DEBUG)
    labels = load_labels()
    load_data(labels)


if __name__ == "__main__":
    main()