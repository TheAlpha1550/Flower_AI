import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from os import listdir
from os.path import join
import numpy as np
from numpy import expand_dims, vstack
import matplotlib.pyplot as plt

IMG_SHAPE = 128

flower_model = load_model("Flower_AI.h5")

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
