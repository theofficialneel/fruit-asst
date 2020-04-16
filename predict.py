import numpy as np
import sys
import json
from datetime import date
from scipy.misc import imread, imresize, imshow
from core_predict import predict

params = np.load("params.npy").item()

def predict_fruit(img_name):
    try:
        im = imread(img_name, mode='RGB')
        im = imresize(im, (40,40,3))
        im = im.reshape(40*40*3,1)
    except ValueError:
        print("Error reading ", img_name)

    im = im/255
    yp = predict(im, params)
    index = np.where(yp==1)[0][0]
    return index

def disp_facts(index):
    with open('./config/info.json', 'r') as f:
        i = json.load(f)
    
    fruit = i[str(index)]
    print("You have consumed", fruit["name"])
    print("Added new fruit to chart on", date.today())
    print(json.dumps(fruit, indent = 4, sort_keys=True))

while 1:
    img = input("\nEnter the fruit-image file location: ")
    if img == "":
        break
    ind = predict_fruit(img)
    disp_facts(ind)
