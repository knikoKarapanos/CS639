import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from keras.models import load_model

import matplotlib.pyplot as plt
import numpy as np

# Configurable Constants
EPOCHS = 10
BATCH_SIZE = 32


def load_data(filename: str) -> "(np.array, np.array)":
    x_data = pickle.load(open(filename + "_X.pickle", "rb"))
    x_data = x_data / 255.0 # Normalize data
    y_data = pickle.load(open(filename + "_y.pickle", "rb"))
    import pdb; pdb.set_trace()
    return (x_data, y_data)

def create_model(input: np.array, num_classes: int) -> Sequential:
    model = Sequential()

    # 3 convolutional layers
    model.add(Conv2D(64, (3, 3), input_shape=input.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 2 Hidden Layers
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))

    model.add(Dense(128))
    model.add(Activation("relu"))

    # Output layer with N neurons for N classes
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    # Compile model
    model.compile(loss="sparse_categorical_crossentropy",
                    optimizer="adam",
                    metrics=["accuracy"])
    
    return model

def train_model(model: Sequential, x_data: np.array, y_data: np.array, epochs: int, batch_size: int) -> Sequential:
    import pdb; pdb.set_trace()
    history = model.fit(x_data, y_data, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    return model

def save_model(model: Sequential, filename: str) -> None:
    model_json = model.to_json()
    with open(filename + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(filename + ".h5")
    model.save(filename + ".model")


if __name__ == "__main__":
    x_data, y_data = load_data("training_data")
    print("Loaded data")
    model = create_model(x_data, 2)
    print("Created model")
    model = train_model(model, x_data, y_data, 10, 32)
    import pdb; pdb.set_trace()
    # print("Trained model")
    # save_model(model, "model")
    # print("Saved model")

    plt.figure(1)
    plt.plot(model.history.history["acc"])
    plt.plot(model.history.history["val_acc"])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.savefig('accuracy.png')

