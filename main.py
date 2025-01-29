# import libraries and load data
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import datasets, layers, models
import random

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
# Normalize pixel values between 0 and 1
training_images = training_images.astype("float32") / 255.0
testing_images = testing_images.astype("float32") / 255.0

# assign class names n' labels
class_names = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

# define the model architecture
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()

# building and training the model (part 1: picking out training and testing data)
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

# # building and training the model (part 2: building the "brain" out of neurons)
# model = models.Sequential()
# model.add(layers.Conv2D(12, (3,3), activation="relu", input_shape=(32,32,3)))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation="relu"))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64, (3,3), activation="relu"))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation="relu"))
# model.add(layers.Dense(10, activation="softmax"))

# # compiling and fitting the model
# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

# # evaluating the model and saving its training
# loss, accuracy = model.evaluate(testing_images, testing_labels)
# print(f"Loss: {loss}")
# print(f"Accuracy: {accuracy}")

# model.save("image_classifier.keras")

# load the model
model = models.load_model("image_classifier.keras")
print("")
# convert the images for overall flexibility
img = cv.imread("C:/Users/yodiw/OneDrive/Pictures/Saved Pictures/horse.jpg", cv.IMREAD_COLOR)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

# allows the program to read each image for the brightest activation out of the end neurons, or the 'match'
prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)

# makes a random number that corresponds to a statement - and makes its guess, too!
random_num = random.randint(1, 5)
if random_num == 1:
    print(f"I would predict this to be a {class_names[index]}.")
elif random_num == 2:
    print(f"Well, I'll be. I'd say that's a nice {class_names[index]}.")
elif random_num == 3:
    print(f"Hmm...this WAS in my training...aha! It's a {class_names[index]}!")
elif random_num == 4:
    print(f"Oh, hello there Mr. {class_names[index]}! How's your day?")
elif random_num == 5:
    print(f"I don't mean to be a nerd, but according to my neural network, that's a {class_names[index]}. Not trying to be a nerd, though.")