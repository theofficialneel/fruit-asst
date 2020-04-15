import numpy as np
import sys
from scipy.misc import imread, imresize, imshow
from core_predict import predict

op = ["mango", "green_apple", "orange", "pear"]
params = np.load("params.npy").item()
img_name = str(sys.argv[1])

try:
    im = imread(img_name, mode='RGB')
    im = imresize(im, (40,40,3))
    imshow(im)
    im = im.reshape(40*40*3,1)
except ValueError:
    print("Error reading ", img_name)

im = im/255
yp = predict(im, params)
index = np.where(yp==1)[0][0]
print("Image is :", op[index])