import os
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, AveragePooling2D
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
    min_size = 16
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
    return validated, mser_areas


INPUT_PATH = "Input/0.jpg"
TRAIN_PATH = "Data/Train/"
NEGATIVE_PATH = "Data/"

image = cv2.imread(INPUT_PATH)
img_test = image.copy()

valid_windows, pure_mser = get_valid_mser_windows(image)

# hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in pure_mser]
# cv2.polylines(image, hulls, 1, (0, 255, 0))
# print(len(pure_mser))
# cv2.imshow('Pure MSER', image)
# cv2.imwrite("pure_mser.bmp", np.float32(image))

for window in valid_windows:
    cv2.rectangle(img_test, (window[0], window[1]), (window[2], window[3]), (0, 255, 0), 1)
cv2.imshow("Validated areas", img_test)
# cv2.imwrite('validated_mser.jpg', cv2.UMat(img_test))
# print(len(valid_windows))

window_area = []  # КООРДИНАТЫ
window_inputs = []  # ОБЛАСТИ

for window in valid_windows:
    window_area.append(window)
    window_inputs.append(image[window[1]:window[3], window[0]:window[2]])  # ОБРЕЗКА ИЗОБРАЖЕНИЯ ПО КООРДИНАТАМ

inputImages32 = np.array([transform.resize(image, (32, 32)) for image in window_inputs])
# for image in range(15):
#     print(inputImages32[image].shape)
#     cv2.imshow(str(image), inputImages32[image])
path = 'G:/PyProjects/OldNet/Data/Train/Test_false'
for image in range(len(inputImages32)):
    frame_normed = 255 * (inputImages32[image] - inputImages32[image].min()) / (inputImages32[image].max() - inputImages32[image].min())
    frame_normed = np.array(frame_normed, np.int)
    cv2.imwrite(os.path.join(path, str(image) + ".jpg"), frame_normed)


print(inputImages32.shape)







true_test_images = []
true_test_images_lables = []
for image in os.listdir(TRAIN_PATH + '/Test_true'):
    true_test_images.append(cv2.cvtColor(cv2.imread(TRAIN_PATH + "/Test_true/" + image), cv2.COLOR_BGR2RGB))

    true_test_images_lables.append([1, 0])
true_test_images = np.array([transform.resize(image, (32, 32)) for image in true_test_images])
print(true_test_images.shape)
print(len(true_test_images_lables))

false_test_images = []
false_test_images_lables = []
for image in os.listdir(TRAIN_PATH + "/Test_false"):
    false_test_images.append(cv2.cvtColor(cv2.imread(TRAIN_PATH + "/Test_false/" + image), cv2.COLOR_BGR2RGB))
    false_test_images_lables.append([0, 1])
false_test_images = np.array([transform.resize(image, (32, 32)) for image in false_test_images])
print(false_test_images.shape)
print(len(false_test_images_lables))

fin_test = np.concatenate((true_test_images, false_test_images), axis=0)
fin_test_lables = np.array(true_test_images_lables + false_test_images_lables)
print(fin_test.shape)

indices_test = np.arange(fin_test.shape[0])
np.random.shuffle(indices_test)
fin_test = fin_test[indices_test]
fin_test_lables = fin_test_lables[indices_test]





true_images = []
true_images_lables = []
for image in os.listdir(TRAIN_PATH + '/13'):
    true_images.append(cv2.cvtColor(cv2.imread(TRAIN_PATH + "/13/" + image), cv2.COLOR_BGR2RGB))
    true_images_lables.append([1, 0])
true_images = np.array([transform.resize(image, (32, 32)) for image in true_images])
print(true_images.shape)
print(len(true_images_lables))

false_images = []
false_images_lables = []
for image in os.listdir(TRAIN_PATH + "false_mser"):
    false_images.append(cv2.cvtColor(cv2.imread(TRAIN_PATH + "false_mser/" + image), cv2.COLOR_BGR2RGB))
    false_images_lables.append([0, 1])
false_images = np.array([transform.resize(image, (32, 32)) for image in false_images])
print(false_images.shape)
print(len(false_images_lables))

fin = np.concatenate((true_images, false_images), axis=0)
fin_lables = np.array(true_images_lables + false_images_lables)
print(fin.shape)

indices = np.arange(fin.shape[0])
np.random.shuffle(indices)
fin = fin[indices]
fin_lables = fin_lables[indices]

fin = fin / 255
fin_test = fin_test / 255

plt.figure(figsize=(10, 5))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(fin[i], cmap=plt.cm.binary)

plt.show()
cv2.waitKey(0)
model = keras.Sequential([
    Conv2D(192, (5, 5), padding='same', activation='relu', input_shape=(32, 32, 3)),
    Conv2D(160, (1, 1), padding='same', activation='relu', groups=1),
    Conv2D(96, (1, 1), padding='same', activation='relu', groups=1),
    MaxPooling2D((3, 3), strides=2),
    Dropout(0.5),

    Conv2D(192, (5, 5), padding='same', activation='relu'),
    Conv2D(192, (1, 1), padding='same', activation='relu', groups=1),
    Conv2D(192, (1, 1), padding='same', activation='relu', groups=1),
    AveragePooling2D((3, 3), strides=2),
    Dropout(0.5),

    Conv2D(192, (3, 3), padding='same', activation='relu'),
    Conv2D(192, (1, 1), padding='same', activation='relu', groups=1),
    Conv2D(2, (1, 1), padding='same', activation='relu', groups=1),
    AveragePooling2D((8, 8), strides=1, padding='same'),
    Flatten(),
    Dense(2, activation="softmax")
])
model.compile(
    optimizer="adam",
    #optimizer=tf.keras.optimizers.SGD(learning_rate=0.000001, momentum=0.9, nesterov=True),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

print(fin_lables.shape)
print(fin[0].shape)

hist = model.fit(fin,
                 fin_lables,
                 batch_size=32,
                 epochs=5,
                 validation_split=0.2)
model.evaluate(fin_test, fin_test_lables)  # Тестовая выборка







plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.show()


cv2.waitKey(0)
