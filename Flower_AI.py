import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Rescaling, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import get_file

import os
from os.path import join, dirname

import matplotlib.pyplot as plt

# Use "dark mode" for graphs.
plt.style.use("dark_background")

# Function to plot metrics of the trained model.
def plot_metric(model_history, metric):

    # Save train and validation metrics from training history.
    train_metric = model_history.history[metric]
    validation_metric = model_history.history["val_" + metric]

    # Epochs is sequence of integers from 1 to the amount of
    epochs = range(1, len(train_metric) + 1)

    # Plots the training and validation graphs separately.
    # x - axis: amount of epochs as a sequence of intervals
    # y - axis:
    plt.plot(epochs, train_metric)
    plt.plot(epochs, validation_metric)

    # Capitalize each metric for legibility.
    metric = metric.capitalize()

    # Initialize proper labels for each plot.
    plt.title("Training and Validation " + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["Train_" + metric, "Validation_" + metric])

    # Display plots.
    plt.show()


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


# Data Loading

# URL of the source of the file that contains the dataset.
_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

# Obtain the compressed file and extract its contents.
zip_file = get_file(origin = _URL, fname = "flower_photos.tgz", extract = True)

# Create a base path for the dataset that joins the directory of the compressed file and creates a new directory "flower_photos".
base_dir = join(dirname(zip_file), "flower_photos")

# Hyperparameters.
BATCH_SIZE = 32
IMG_SHAPE = 128
EPOCHS = 150

# Create Train and Validation datasets.
train_dataset = image_dataset_from_directory\
(
    # Indicate the directory which hold the training images.
    directory = base_dir,
    # Value between 0 and 1 to indicate percentage of amount to use for validation.
    validation_split = 0.1,
    # Indicate the purpose of this dataset.
    subset = "training",
    # Seed to randomize shuffling.
    seed = 123,
    # Indicate dimensions to resize images in datasets.
    image_size = (IMG_SHAPE, IMG_SHAPE),
    # Indicate batch size.
    batch_size = BATCH_SIZE
)

validation_dataset = image_dataset_from_directory\
(
    # Indicate the directory which hold the training images.
    directory = base_dir,
    # Value between 0 and 1 to indicate percentage of amount to use for validation.
    validation_split = 0.1,
    # Indicate what the purpose of this dataset.
    subset = "validation",
    # Seed to randomize shuffling.
    seed = 123,
    # Indicate dimensions to resize images in datasets.
    image_size = (IMG_SHAPE, IMG_SHAPE),
    # Indicate batch size.
    batch_size = BATCH_SIZE
)

# The classes of flowers in the dataset.
# These correspond to the directory names in alphabetical order.
class_names = train_dataset.class_names
classAmount = len(class_names)
# print(class_names)

# (32, 128, 128, 3) tensor
# (32,) tensor
# for image_batch, labels_batch in train_dataset:
#    print(image_batch.shape)
#    print(labels_batch.shape)
#    break

# Configure dataset for performance.

# Dataset.cache keeps the images in memory after they're loaded off disk during the first epoch.
# This will ensure the dataset does not become a bottleneck while training your model.
# If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache.
# Dataset.prefetch overlaps data preprocessing and model execution while training.
AUTOTUNE = tensorflow.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size = AUTOTUNE)

# Create model.
model = Sequential\
([
    # BUG! caused by TF2.10!. Source: https://stackoverflow.com/a/73774497
    # RandomFlip("horizontal", input_shape = (IMG_SHAPE, IMG_SHAPE, 3)),
    # RandomRotation(0.2),
    # RandomZoom(0.2),
    # Rescaling(1./255),

    Conv2D(filters = 16, kernel_size = 7, padding = "same", activation = relu),
    MaxPooling2D(pool_size = 2),
    Conv2D(filters = 32, kernel_size = 7, padding = "same", activation = relu),
    MaxPooling2D(pool_size = 2),
    Conv2D(filters = 64, kernel_size = 5, padding = "same", activation = relu),
    MaxPooling2D(pool_size = 2),
    Conv2D(filters = 128, kernel_size = 5, padding = "same", activation = relu),
    MaxPooling2D(pool_size = 2),
    Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = relu),
    MaxPooling2D(pool_size = 2),
    Conv2D(filters = 512, kernel_size = 3, padding = "valid", activation = relu),
    MaxPooling2D(pool_size = 2),

    Flatten(),

    Dropout(0.2),
    Dense(256, activation = relu),
    Dropout(0.2),
    Dense(128, activation = relu),
    Dropout(0.2),
    Dense(64, activation = relu),
    Dropout(0.2),
    Dense(32, activation = relu),
    Dropout(0.2),
    Dense(16, activation= relu),
    Dropout(0.2),
    Dense(classAmount, activation = softmax)
])

# Compile the model.
model.compile(optimizer = Adam(), loss = SparseCategoricalCrossentropy(), metrics = ["accuracy"])

# Set parameters to allow training to stop early.
earlyStopping = EarlyStopping(patience = 50, restore_best_weights = True)

# Train model.
history = model.fit\
(
        x = train_dataset,
        validation_data = validation_dataset,
        epochs = EPOCHS,
        callbacks = [earlyStopping]
)

# Plot metrics of trained model.
plot_metric(history, "loss")
plot_metric(history, "accuracy")

# Print summary of the model.
print(model.summary())

# Save model.
model.save("Flower_AI.h5")