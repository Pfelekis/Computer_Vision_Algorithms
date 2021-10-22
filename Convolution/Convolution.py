import os
import PIL
import requests
import math

import numpy as np
import matplotlib.pyplot as plt

### Download example image

# Image location and path
#url = "https://github.com/ShawnHymel/computer-vision-with-embedded-machine-learning/raw/master/2.1.4%20-%20Project%20-%20Convolution%20and%20Pooling/resistor.png"
#img_path = os.path.join("/Convolution", "dog.png")

# Download image
#resp = requests.get(url)

# Write image to file
#with open(img_path, 'wb') as f:
#  f.write(resp.content)


  ### Open and view image

# Use PIL to open the image and convert it to grayscale
img = PIL.Image.open("dog.png")
img = img.convert('L')

# Convert image to Numpy array
img = np.asarray(img)

# Show dimensions and view array as image
print(img.shape)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()
### Convolution function

def convolve(img, kernel, stride):

  # Compute dimensions of output image
  out_height = math.floor((img.shape[0] - kernel.shape[0]) / stride) + 1
  out_width = math.floor((img.shape[1] - kernel.shape[1]) / stride) + 1
  
  # Create blank output image
  convolved_img = np.zeros((out_height, out_width))

  # >>> ENTER YOUR CODE HERE <<<
  
  # Loop through each pixel in the output array. Note that this is not the most efficient way of
  # doing convolution, but it provides some insights into what's going on.
  for i in np.arange(0, out_height):
    for j in np.arange(0, out_width):

      # Set a temporary variable to 0
      accumulator = 0

      # Do element-wise multiplication and sum the result over the window/kernel
      for m in np.arange(0, kernel.shape[0]):
        for n in np.arange(0, kernel.shape[1]):
          accumulator += img[(stride * i) + m, (stride * j) + n] * kernel[m, n]

      # Set output image pixel to accumulator value
      convolved_img[i, j] = accumulator

  # Round all elements, convert to integers, and clamp to values between 0 and 255
  convolved_img = np.rint(convolved_img).astype(int)
  convolved_img = np.clip(convolved_img, 0, 255)

  return convolved_img



  ### Test 1: Gaussian blur filter

# Define kernel
kernel = np.array([[1/16, 2/16, 1/16],
                   [2/16, 4/16, 2/16],
                   [1/16, 2/16, 1/16]])

# Call your convolve function (with a stride of 1)
out_img = convolve(img, kernel, 1)

# Show dimensions and view array as image
print(out_img.shape)
plt.imshow(out_img, cmap='gray', vmin=0, vmax=255)
plt.show()


### Test 2: Edge detection

# Define kernel
kernel = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])

# Call your convolve function (with a stride of 1)
out_img = convolve(img, kernel, 1)

# Show dimensions and view array as image
print(out_img.shape)
plt.imshow(out_img, cmap='gray', vmin=0, vmax=255)
plt.show()


### Test 3: Sharpen with stride > 1

# Define kernel
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

# Call your convolve function (with a stride of 2)
out_img = convolve(img, kernel, 2)

# Show dimensions and view array as image
print(out_img.shape)
plt.imshow(out_img, cmap='gray', vmin=0, vmax=255)
plt.show()