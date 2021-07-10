from nn import DenseLayer
import numpy as np
from PIL import Image
import glob
import random
import getpass

#my fully connected layer has 784 input neurons, 40 hidden neurons and 3 output neurons -> the numbers to classify are 0,1 and 2
w = [np.random.uniform(-0.5, 0.5, (784,40)),np.random.uniform(-0.5, 0.5, (40,40)),np.random.uniform(-0.5, 0.5, (40,3))]
nn = DenseLayer([784, 40, 40, 3], ["TanH", "Sigmoid", "Softmax"], w)
learning_rate = 0.02
epochs = 3
training_data = []
training_data_desired_out = []
test_data = []
test_data_desired_out = []
user = getpass.getuser()

print("Images are being read in")
for file in glob.glob('C:\\Users\\' + user + '\\Desktop\\mnist_png\\testing\\1\\*.png'):
    image = Image.open(file)
    rgb = image.convert("RGB")
    width, height = image.size
    out = []
    for c in range(width):
        for d in range(height):
            pixel = rgb.getpixel((c, d))
            out.append(pixel[0]/255)
    test_data.append([out])
    test_data_desired_out.append([0,1,0])
for file in glob.glob('C:\\Users\\' + user + '\\Desktop\\mnist_png\\testing\\0\\*.png'):
    image = Image.open(file)
    rgb = image.convert("RGB")
    width, height = image.size
    out = []
    for c in range(width):
        for d in range(height):
            pixel = rgb.getpixel((c, d))
            out.append(pixel[0]/255)
    test_data.append([out])
    test_data_desired_out.append([1,0,0])
for file in glob.glob('C:\\Users\\' + user +'\\Desktop\\mnist_png\\testing\\2\\*.png'):
    image = Image.open(file)
    rgb = image.convert("RGB")
    width, height = image.size
    out = []
    for c in range(width):
        for d in range(height):
            pixel = rgb.getpixel((c, d))
            out.append(pixel[0]/255)
    test_data.append([out])
    test_data_desired_out.append([0,0,1])
for file in glob.glob('C:\\Users\\' + user + '\\Desktop\\mnist_png\\training\\0\\*.png'):
    image = Image.open(file)
    rgb = image.convert("RGB")
    width, height = image.size
    out = []
    for c in range(width):
        for d in range(height):
            pixel = rgb.getpixel((c, d))
            out.append(pixel[0]/255)
    training_data.append([out])
    training_data_desired_out.append([1,0,0])

for file in glob.glob('C:\\Users\\' + user + '\\Desktop\\mnist_png\\training\\1\\*.png'):
    image = Image.open(file)
    rgb = image.convert("RGB")
    width, height = image.size
    out = []
    for c in range(width):
        for d in range(height):
            pixel = rgb.getpixel((c, d))
            out.append(pixel[0]/255)
    training_data.append([out])
    training_data_desired_out.append([0,1,0])

for file in glob.glob('C:\\Users\\' + user + '\\Desktop\\mnist_png\\training\\2\\*.png'):
    image = Image.open(file)
    rgb = image.convert("RGB")
    width, height = image.size
    out = []
    for c in range(width):
        for d in range(height):
            pixel = rgb.getpixel((c, d))
            out.append(pixel[0]/255)
    training_data.append([out])
    training_data_desired_out.append([0,0,1])

print("Images were read in")

def accuracy():
    count = 0
    for i in range(len(test_data)):
        x = nn.forward(test_data[i])
        x_max = np.amax(x)
        if x_max == x[0][0]:
            x_max_i = 0
        if x_max == x[0][1]:
            x_max_i = 1
        if x_max == x[0][2]:
            x_max_i = 2
    
        y_max_i= test_data_desired_out[i].index(max(test_data_desired_out[i]))
        if x_max_i == y_max_i:
            count += 1
    return (count / len(test_data)) * 100

c = list(zip(training_data, training_data_desired_out))
random.shuffle(c)
training_data, training_data_desired_out = zip(*c)
training_data = list(training_data)
training_data_desired_out = list(training_data_desired_out)

d = list(zip(test_data, test_data_desired_out))
random.shuffle(d)
test_data, test_data_desired_out = zip(*d)
test_data = list(test_data)
test_data_desired_out = list(test_data_desired_out)

print("Accuracy before learning: " + str(accuracy()) + "%")
print("Now it's time to learn!")
for i in range(epochs):
    for j in range(len(training_data)):
        x = nn.forward(training_data[j])
        nn.backward(training_data_desired_out[j], learning_rate)
    print("Epoch done: " + str(i+1))
print("Accuracy after learning: " + str(accuracy()) + "%")
