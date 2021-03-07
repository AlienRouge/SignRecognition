import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import transform, io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_valid_mser_windows(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    padding = 5
    mser = cv2.MSER_create(_delta=1)
    mser_areas, _ = mser.detectRegions(img)

    mser_windows = []
    for area in mser_areas:
        df_area = pd.DataFrame(area)
        xmin, ymin = df_area.min()
        xmax, ymax = df_area.max()

        xmin -= padding
        ymin -= padding
        xmax += padding
        ymax += padding

        if xmin >= 0 and ymin >= 0 and xmax <= img.shape[1] and ymax <= img.shape[0]:
            mser_windows.append((xmin, ymin, xmax, ymax))

    validated = []
    min_size = 50
    max_size = 200
    max_proportion = 1.2

    # xmin ymin xmax ymax
    for win in mser_windows:
        size_x = float(win[2] - win[0])
        size_y = float(win[3] - win[1])
        if (size_x < min_size) or (size_y < min_size) or (size_x > max_size) or (size_y > max_size) or \
                (size_x / size_y > max_proportion) or (size_y / size_x > max_proportion):
            continue
        validated.append(win)

    window_area = []  # КООРДИНАТЫ
    window_inputs = []  # ОБЛАСТИ
    for window in validated:
        window_area.append(window)
        window_inputs.append(image[window[1]:window[3], window[0]:window[2]])  # ОБРЕЗКА ИЗОБРАЖЕНИЯ ПО КООРДИНАТАМ
    return window_area, window_inputs


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


INPUT_PATH = "Input/5.jpg"
TRAIN_PATH = "Data/Train/"
NEGATIVE_PATH = "Data/Train/false"

input_images = load_images_from_folder(NEGATIVE_PATH)

mser_false_list = list()
for image in input_images:
    _, false_mser = get_valid_mser_windows(image)
    mser_false_list = mser_false_list + false_mser
    print(type(false_mser))
    print(len(mser_false_list))


path = 'G:/PyProjects/OldNet/Data/Train/false_mser'
for image in range(len(mser_false_list)):
    cv2.imwrite(os.path.join(path, str(image)+".jpg"), mser_false_list[image])


# image = cv2.imread(NEGATIVE_PATH)
# img_test = image.copy()
#
# valid_windows, pure_mser = get_valid_mser_windows(image)
#
# for window in valid_windows:
#     cv2.rectangle(img_test, (window[0], window[1]), (window[2], window[3]), (0, 255, 0), 1)
# cv2.imshow("Validated areas", img_test)
#
# window_area = []  # КООРДИНАТЫ
# window_inputs = []  # ОБЛАСТИ
#
# for window in valid_windows:
#     window_area.append(window)
#     window_inputs.append(image[window[1]:window[3], window[0]:window[2]])  # ОБРЕЗКА ИЗОБРАЖЕНИЯ ПО КООРДИНАТАМ
