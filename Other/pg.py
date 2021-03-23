import cv2
import pandas as pd
import math
import os
import sign_recognition




path="Input/"

filelist = [f for f in os.listdir(path)]

image = cv2.imread("Input/Negative/10.png")
image = sign_recognition.contrast_stabilization(image)

tuples, areas = sign_recognition.get_valid_mser_windows(image)
print(len(tuples))
mser_path = 'G:/PyProjects/OldNet/Data/MSER/'
filelist = [f for f in os.listdir(mser_path)]
for f in filelist:
    os.remove(os.path.join(mser_path, f))
for image in range(len(areas)):
    cv2.imwrite(mser_path + str(tuples[image]) + ".png", areas[image])

cv2.waitKey(0)
