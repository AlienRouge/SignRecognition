import os
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, AveragePooling2D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pydot

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def get_valid_mser_windows(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

    validated_areas = []
    min_area_size = 16
    max_area_size = 200
    max_proportion = 1.2

    # КООРДИНАТЫ (xmin ymin xmax ymax)
    for win in mser_windows:
        size_x = float(win[2] - win[0])
        size_y = float(win[3] - win[1])
        if (size_x < min_area_size) or (size_y < min_area_size) or (size_x > max_area_size) or (
                size_y > max_area_size) or \
                (size_x / size_y > max_proportion) or (size_y / size_x > max_proportion):
            continue
        validated_areas.append(win)
    # КОНТУРЫ
    img_test = image.copy()
    for area in validated_areas:
        cv2.rectangle(img_test, (area[0], area[1]), (area[2], area[3]), (0, 255, 0), 1)
    cv2.imshow("Validated areas", img_test)

    # ИЗОБРАЖЕНИЯ
    window_tuples = []  # КООРДИНАТЫ
    window_areas = []  # ОБЛАСТИ
    for window in validated_areas:
        window_tuples.append(window)
        window_areas.append(image[window[1]:window[3], window[0]:window[2]])  # ОБРЕЗКА ИЗОБРАЖЕНИЯ ПО КООРДИНАТАМ

    return window_tuples, window_areas


def get_images_from_folder(path, is_true):
    images = []
    lables = []
    for _image in os.listdir(path):
        images.append(cv2.imread(path + _image))
        lables.append([1, 0] if is_true == 1 else [0, 1])
    return images, lables


def create_and_shuffle_dataset(true_x, false_x, true_y, false_y):
    x = np.concatenate((true_x, false_x), axis=0)
    y = np.array(true_y + false_y)
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    return x, y

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


INPUT_PATH = "Input/0.jpg"
TRAIN_PATH = "Data/Train/"
TEST_PATH = "Data/Test/"
NEGATIVE_PATH = "Data/"

input_image = cv2.imread(INPUT_PATH)
mser_tuples, mser_windows = get_valid_mser_windows(input_image)

inputImages32 = np.array([cv2.resize(image, (32, 32)) for image in mser_windows])

mser_path = 'G:/PyProjects/OldNet/Data/MSER'
for image in range(len(inputImages32)):
    cv2.imwrite(os.path.join(mser_path, str(image) + ".jpg"), inputImages32[image])
print("MSER32 AREAS:", inputImages32.shape)

true_test_x, true_test_y = get_images_from_folder(TEST_PATH + "1/", 1)
true_test_x = np.array([cv2.resize(image, (32, 32)) for image in true_test_x])
print("[True test images] -", true_test_x.shape, "[True test lables:] -", len(true_test_y), "[B] -", true_test_y[0])

false_test_x, false_test_y = get_images_from_folder(TEST_PATH + "0/", 0)
false_test_x = np.array([cv2.resize(image, (32, 32)) for image in false_test_x])
print("[False test images] -", false_test_x.shape, "[False test lables] -", len(false_test_y), "[B] -", false_test_y[0])

test_x, test_y = create_and_shuffle_dataset(true_test_x, false_test_x, true_test_y, false_test_y)


num_skipped = 0
for folder_name in ("1", "0"):
    folder_path = os.path.join(TEST_PATH, folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print("Deleted %d images" % num_skipped)

image_size = (32, 32)
batch_size = 32

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_PATH,
    labels="inferred",
    label_mode="binary",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=image_size,
    shuffle=True,
    validation_split=None,
    subset=None,
    interpolation="bilinear"
)

plt.figure(figsize=(10, 10))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
plt.show()

# num_skipped = 0
# for folder_name in ("1", "0"):
#     folder_path = os.path.join(TRAIN_PATH, folder_name)
#     for fname in os.listdir(folder_path):
#         fpath = os.path.join(folder_path, fname)
#         try:
#             fobj = open(fpath, "rb")
#             is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
#         finally:
#             fobj.close()
#
#         if not is_jfif:
#             num_skipped += 1
#             # Delete corrupted image
#             os.remove(fpath)
#
# print("Deleted %d images" % num_skipped)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_PATH,
    labels="inferred",
    label_mode="binary",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=image_size,
    shuffle=True,
    seed=1688,
    validation_split=0.2,
    subset="training",
    interpolation="bilinear"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_PATH,
    labels="inferred",
    label_mode="binary",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=image_size,
    shuffle=True,
    seed=1688,
    validation_split=0.2,
    subset="validation",
    interpolation="bilinear"
)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
plt.show()

# train_ds = train_ds.prefetch(buffer_size=32)
# val_ds = val_ds.prefetch(buffer_size=32)
#
# model = make_model(input_shape=image_size + (3,), num_classes=2)
# #keras.utils.plot_model(model, show_shapes=True)
#
# epochs = 20
#
# callbacks = [
#     keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
# ]
# model.compile(
#     optimizer=keras.optimizers.Adam(1e-3),
#     loss="binary_crossentropy",
#     metrics=["accuracy"],
# )
# hist = model.fit(
#     train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
# )

model = tf.keras.models.load_model('save_at_17.h5')
# model.evaluate(test_ds)  # Тестовая выборка
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.show()

predict =model.predict(test_x)
for x in range(10):
    print(predict[x], "-", x)
    print(test_y[x])
    cv2.imshow(str(x), test_x[x])

# img = keras.preprocessing.image.load_img(
#     "Data/Test/1/139.jpg", target_size=image_size
# )
#
# img_array = keras.preprocessing.image.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0)  # Create batch axis
# cv2.imshow("xyz", cv2.imread("Data/Test/1/139.jpg"))
#
# predictions = model.predict(img_array)
# print(predictions)
#
#
# img = keras.preprocessing.image.load_img(
#     "Data/Test/0/7.jpg", target_size=image_size
# )
#
# img_array = keras.preprocessing.image.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0)  # Create batch axis
# cv2.imshow("xyz", cv2.imread("Data/Test/0/7.jpg"))
#
# predictions = model.predict(img_array)
# print(predictions)

cv2.waitKey(0)
# pred = model.predict(test_x)
#
# for x in range(10):
#     cv2.imshow(str(x), test_x[x])
#     print(test_y[x])
#     score = pred[x]
#     print(
#         "This image is %.2f percent cat and %.2f percent dog."
#         % (100 * (1 - score), 100 * score)
#     )
#     print(pred[x])
#
# cv2.waitKey(0)
# pred = np.argmax(pred, axis=1)
#
# print(pred.shape)
#
# print(pred[:20])
# print(test_y[:20])
#
# # Выделение неверных вариантов
# mask = pred == test_y
# # print(mask[:10])
#
# x_false = test_x[~mask]
# y_false = test_x[~mask]
# p_false = pred[~mask]
#
# print(x_false.shape)
#
# # Вывод первых 5 неверных результатов
# for i in range(5):
#     print("Значение сети: " + str(p_false[i]))
#     plt.imshow(x_false[i], cmap=plt.cm.binary)
#     plt.show()









#
#
#
#
#
# true_images = []
# true_images_lables = []
# for image in os.listdir(TRAIN_PATH + '/13'):
#     true_images.append(cv2.cvtColor(cv2.imread(TRAIN_PATH + "/13/" + image), cv2.COLOR_BGR2RGB))
#     true_images_lables.append([1, 0])
# true_images = np.array([transform.resize(image, (32, 32)) for image in true_images])
# print(true_images.shape)
# print(len(true_images_lables))
#
# false_images = []
# false_images_lables = []
# for image in os.listdir(TRAIN_PATH + "false_mser"):
#     false_images.append(cv2.cvtColor(cv2.imread(TRAIN_PATH + "false_mser/" + image), cv2.COLOR_BGR2RGB))
#     false_images_lables.append([0, 1])
# false_images = np.array([transform.resize(image, (32, 32)) for image in false_images])
# print(false_images.shape)
# print(len(false_images_lables))
#
# fin = np.concatenate((true_images, false_images), axis=0)
# fin_lables = np.array(true_images_lables + false_images_lables)
# print(fin.shape)
#
# indices = np.arange(fin.shape[0])
# np.random.shuffle(indices)
# fin = fin[indices]
# fin_lables = fin_lables[indices]
#
# fin = fin / 255
# fin_test = fin_test / 255
#
# plt.figure(figsize=(10, 5))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(fin[i], cmap=plt.cm.binary)
#
# plt.show()
# cv2.waitKey(0)
# model = keras.Sequential([
#     Conv2D(192, (5, 5), padding='same', activation='relu', input_shape=(32, 32, 3)),
#     Conv2D(160, (1, 1), padding='same', activation='relu', groups=1),
#     Conv2D(96, (1, 1), padding='same', activation='relu', groups=1),
#     MaxPooling2D((3, 3), strides=2),
#     Dropout(0.5),
#
#     Conv2D(192, (5, 5), padding='same', activation='relu'),
#     Conv2D(192, (1, 1), padding='same', activation='relu', groups=1),
#     Conv2D(192, (1, 1), padding='same', activation='relu', groups=1),
#     AveragePooling2D((3, 3), strides=2),
#     Dropout(0.5),
#
#     Conv2D(192, (3, 3), padding='same', activation='relu'),
#     Conv2D(192, (1, 1), padding='same', activation='relu', groups=1),
#     Conv2D(2, (1, 1), padding='same', activation='relu', groups=1),
#     AveragePooling2D((8, 8), strides=1, padding='same'),
#     Flatten(),
#     Dense(2, activation="softmax")
# ])
# model.compile(
#     coptimizer="adam",
#     #optimizer=tf.keras.optimizers.SGD(learning_rate=0.000001, momentum=0.9, nesterov=True),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# print(model.summary())
#
# print(fin_lables.shape)
# print(fin[0].shape)
#
# hist = model.fit(fin,
#                  fin_lables,
#                  batch_size=32,
#                  epochs=5,
#                  validation_split=0.2)
# model.evaluate(fin_test, fin_test_lables)  # Тестовая выборка
#
#
#
#
#
#
#
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.show()
#
#
# cv2.waitKey(0)
