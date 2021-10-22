import os
import PIL
import requests
import math

import numpy as np
import matplotlib.pyplot as plt

from Convolution import convolve

### Convolution function

def maxpooling(img, pool_height, pool_width):

  # Set stride amounts
  stride_y = pool_height
  stride_x = pool_width

  # Compute dimensions of output image
  out_height = math.floor((img.shape[0] - kernel.shape[0]) / stride_y) + 1
  out_width = math.floor((img.shape[1] - kernel.shape[1]) / stride_x) + 1
  
  # Create blank output image
  pooled_img = np.zeros((out_height, out_width))

  # >>> ENTER YOUR CODE HERE <<<

  # Loop through each pixel in the output array. Note that this is not the most efficient way of
  # doing convolution, but it provides some insights into what's going on.
  for i in np.arange(0, out_height):
    for j in np.arange(0, out_width):

      # Set output to the value of the first element in the window
      out_val = img[(stride_y * i), (stride_x * j)]

      # Look through each element in the window to find the max value
      for m in np.arange(0, pool_height):
        for n in np.arange(0, pool_width):
          out_val = max(out_val, img[(stride_y * i) + m, (stride_x * j) + n])

      # Set element in output array to max value
      pooled_img[i, j] = out_val

  # Round all elements, convert to integers, and clamp to values between 0 and 255
  pooled_img = np.rint(pooled_img).astype(int)
  pooled_img = np.clip(pooled_img, 0, 255)

  return pooled_img


  ### Test 1: Max pool original image with pool size of (2, 3)


img = PIL.Image.open("dog.png")
img = img.convert('L')

# Convert image to Numpy array
img = np.asarray(img)


kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])


# Call your pooling function (with pool_size=(2, 3))
out_img = maxpooling(img, 2, 3)

# Show dimensions and view array as image
print(out_img.shape)
plt.imshow(out_img, cmap='gray', vmin=0, vmax=255)


### Test 2: Detect edges and pool

# Define kernel
kernel = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])

# Call your convolve function (with a stride of 1)
convolved_img = convolve(img, kernel, 1)

# Call your pooling function (with pool_size=(2, 2))
out_img = maxpooling(convolved_img, 2, 2)

# Show dimensions and view array as image
print(out_img.shape)
plt.imshow(out_img, cmap='gray', vmin=0, vmax=255)
plt.show()