import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from os import listdir
from os.path import join
import numpy as np
from numpy import expand_dims, vstack
import matplotlib.pyplot as plt

'''
plt.figure(figsize=(20, 10))
for images, labels in train_dataset.take(1):
    for i in range(15):
        ax = plt.subplot(3, 5, i + 1)

        img_array = images[i].numpy().astype("uint8")
        prediction = model.predict(np.array([img_array]))
        prediction_name = class_names[np.argmax(prediction)]
        real_name = class_names[np.argmax(labels[i])]

        plt.imshow(img_array)
        if prediction_name == real_name:
            plt.title(f'real: {real_name}\npred:{prediction_name}', fontdict={'color': 'g'})
        else:
            plt.title(f'real: {real_name}\npred:{prediction_name}', fontdict={'color': 'r'})

        plt.axis("off")

'''

IMG_SHAPE = 128

# Load the model.
flower_model = load_model("Flower_AI.h5")

# Print a summary of the model.
#print(flower_model.summary())

test_directory = "Flowers"

classes = np.array(["roses", "daisy", "dandelion", "sunflowers", "tulips"])
flowers = np.array(["Rose", "Daisy", "Dandelion", "Sunflower", "Tulip"])

classes.sort()
flowers.sort()

# Iterate over my image files in "Flowers" directory.
for filename in listdir(test_directory):

    path = join(test_directory, filename)
    currentImage = load_img(path, target_size=(IMG_SHAPE, IMG_SHAPE))
    x = img_to_array(currentImage)
    x = expand_dims(x, axis=0)

    # image_tensor = vstack([x])

    predictions = flower_model.predict(x)

    plt.figure()
    plt.title(label = filename + str(" is a " + flowers[np.argmax(predictions[0])] + '.'), color = "blue")
    plt.imshow(currentImage)

    print(filename + ": " + str(predictions))

    print("This image is a " + str(classes[np.argmax(predictions[0])]) + " with a " + str(100 * np.max(predictions[0])) + " score.")

    plt.show()