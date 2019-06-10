import cv2 as cv
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("image_input_1",help="Input image 1 name with extension")
parser.add_argument("image_input_2",  help="Input image 2 name with extension")
args = parser.parse_args()

im = cv.imread(args.image_input_1, cv.IMREAD_COLOR)
width, height, channels = im.shape
TP = width * height
lower_blue = np.array([25,50,35])
upper_blue = np.array([60,100,70])
hsv_img = cv.cvtColor(im,cv.COLOR_BGR2HSV)
mask_1 = cv.inRange(im, lower_blue, upper_blue)
n_white_pix = np.sum(mask_1 == 255)
area_1 = round(n_white_pix / TP * 100, 4)
res_1 = cv.bitwise_and(im,im, mask= mask_1)

im = cv.imread(args.image_input_2, cv.IMREAD_COLOR)
width, height, channels = im.shape
TP = width * height
lower_blue = np.array([25,50,35])
upper_blue = np.array([60,100,70])
hsv_img = cv.cvtColor(im,cv.COLOR_BGR2HSV)
mask_2 = cv.inRange(im, lower_blue, upper_blue)
n_white_pix = np.sum(mask_2 == 255)
area_2 = round(n_white_pix / TP * 100, 4)
res_2 = cv.bitwise_and(im,im, mask= mask_2)


difference = area_1 - area_2

img_diff = cv.absdiff(res_1, res_2)


cv.imshow('Quantized',img_diff)
print("Change in coverage:", difference)

if (difference > 0.1800):
    print("Deforestation detected in image")
elif (difference < 0):
    print("Afforestation detected in image")
else: 
    print("No change in forest cover detected in image")

k = cv.waitKey(0) & 0xFF

