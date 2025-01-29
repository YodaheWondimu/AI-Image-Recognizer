# AI-Image-Recognizer
An AI program that recognizes and characterizes images through its neural network - which is basically a brain. Think about that for a minute. Have YOU ever built a brain? No? Well, I can show you how.

Where does every program start? Easy question - the libraries that it needs. It's just got to borrow a few things and be good to go.
```
# import libraries and load data
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import datasets, layers, models
import random
```
Plus, you may also see this, which is there to normalize the pixel values and get the dataset we'll be using good to go. This process pulls the public images we'll be using our for our usage and makes sure that the training and testing images are presented in a format that will make life easier for us (and the brain we're building) later.
```
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
# Normalize pixel values between 0 and 1
training_images = training_images.astype("float32") / 255.0
testing_images = testing_images.astype("float32") / 255.0
```
There are a lot of things in this world, but for the sake of the model, we use only 10 things for the recognizer to recognize. This gets all the names that it'll be using ready!
```
# assign class names n' labels
class_names = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
```
Any good building has a plan involved, of course, and luckily we have one. It's not a bad plan either, just look at this model architecture - making a map of images and names for the AI to train with later.
```
# define the model architecture
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()
```
We're also going to get ready to build and train the model, now that we've reached this point. So far, we've given the AI libraries, some normalized images, some class names and labels and an architecture. It's time to build the model altogether - starting with part 1.
```
# building and training the model (part 1: picking out training and testing data)
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]
```
Previously on: Building and Training the Model, we got sample sizes of training and testing data that the neural netowrk will delight in. It's already going to be happy just thinking about it. Speaking of thinking, here's part 2:
```
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
```
We already have the model, of course, so it doesn't necessarily need to train itself over and over again. Just know that the training session was complete, and thanks to this part (and part 1, what a great series!), where we built the brain together neuron-by-neuron, the session won't have to be redone. It's conveniently saved for us for later, and that later is now! Yes, yes, we're going to delve into the loading of the model and its conversions on the photos for overall flexibility!
```
# load the model
model = models.load_model("image_classifier.keras")
print("")
# convert the images for overall flexibility
img = cv.imread("C:path/to/the/file/horse.jpg", cv.IMREAD_COLOR)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)
```
Time for the model to use what it learned! The piece de resistance, where the AI's neural network checks for the brightest neural activation at the end of the pattern recognition process to find a match with a name! There are even randomized statements every time the AI makes a prediction, just look at that ending!
```
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
```
That's the basics of how an image recognition model works! Now you know how to build a brain, so have fun making things come to life!

Applications after applications are present for an AI recognition model. Whether it's for letting AI play chess with you, or make helpful medical diagnoses along with the professionals, or even find and recognize people and animals in an image, the possibilities are endless for such a next-level model as this. Thank you for reading all of this, and I hope you keep using YOUR neurons in your computer programming journey!

- Yodahe Wondimu, Computer Scientist
