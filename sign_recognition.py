import os
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import models

# 1. Разделить код
# 2. Выделить тренировки
# 3. Попробовать другие виды НС
# 3. Отчистить код
# 4. Залить на гит

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def contrast_stabilization(_image):
    # Converting image to LAB Color model
    lab = cv2.cvtColor(_image, cv2.COLOR_BGR2LAB)
    # Splitting the LAB image to different channels
    l, a, b = cv2.split(lab)
    # Applying CLAHE to L-channe
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    # Merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))
    # Converting image from LAB Color model to RGB model
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def int_r(num):
    num = int(num + (0.5 if num > 0 else -0.5))
    return num


def get_valid_mser_windows(_image):
    img = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
    padding = 8
    mser = cv2.MSER_create(_delta=4)
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
    min_area_size = 30
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
    img_test = _image.copy()
    for area in validated_areas:
        cv2.rectangle(img_test, (area[0], area[1]), (area[2], area[3]), (100, 255, 0), 1)
    cv2.imshow("Validated areas", img_test)

    # ИЗОБРАЖЕНИЯ
    window_tuples = []  # КООРДИНАТЫ
    window_areas = []  # ОБЛАСТИ

    for window in validated_areas:
        window_tuples.append(window)
        window_areas.append(_image[window[1]:window[3], window[0]:window[2]])  # ОБРЕЗКА ИЗОБРАЖЕНИЯ ПО КООРДИНАТАМ

    return window_tuples, window_areas


def get_images_from_folder(path, is_true):
    images = []
    lables = []
    for _image in os.listdir(path):
        images.append(cv2.imread(path + _image))
        lables.append([1, 0] if is_true == 1 else [0, 1])
    return images, lables


def shuffle_dataset(true_x, false_x, true_y, false_y):
    x = np.concatenate((true_x, false_x), axis=0)
    y = np.array(true_y + false_y)
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    return x, y


def create_dataset(true, false):
    true_x, true_y = get_images_from_folder(true, 1)
    true_x = np.array([cv2.resize(_image, (32, 32)) for _image in true_x])
    false_x, false_y = get_images_from_folder(false, 0)
    false_x = np.array([cv2.resize(_image, (32, 32)) for _image in false_x])
    ds_x, ds_y = shuffle_dataset(true_x, false_x, true_y, false_y)
    return ds_x, ds_y



# ВХОДНЫЕ ДАННЫЕ
INPUT_PATH = "Input/1092.jpg"
TRAIN_PATH = "Data/Train/"
TEST_PATH = "Data/Test/"
NEGATIVE_PATH = "Data/"
TRAINING_MODE = False
load_model = 'save_at_30.h5'
epochs = 30
batch_size = 32
acc = 1e-4
image_size = (32, 32)

# ЛОКАЛИЗАЦИЯ ОБЛАСТЕЙ И ПРИВЕДЕНИЕ К ЗАДАННОМУ РАЗМЕРУ
input_image = cv2.imread(INPUT_PATH)
image = input_image.copy()
image = contrast_stabilization(image)
tuples, areas = get_valid_mser_windows(image)
inputImages32 = np.array([cv2.resize(image, (32, 32)) for image in areas])

# СОХРАНЕНИЕ ПОЛУЧЕННЫХ ОБЛАСТЕЙ В ПАПКУ
mser_path = 'G:/PyProjects/OldNet/Data/MSER/'
filelist = [f for f in os.listdir(mser_path)]
for f in filelist:
    os.remove(os.path.join(mser_path, f))
for image in range(len(inputImages32)):
    cv2.imwrite(mser_path + str(tuples[image]) + ".png", areas[image])
print("MSER32 AREAS:", inputImages32.shape)

# ФОРМИРОВАНИЕ ДАТАСЕТА ДЛЯ ТЕСТА(ВРУЧНУЮ)
test_x, test_y = create_dataset(TEST_PATH + "1/", TEST_PATH + "0/")
print("[Test images] -", test_x.shape, "[Test lables:] -", len(test_y), "[B] -", test_y[0])

# ФОРМИРОВАНИЕ ДАТАСЕТА ДЛЯ ТЕСТА(KERAS)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_PATH,
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
    validation_split=None,
    subset=None,
    interpolation="bilinear")

# ВЫВОД ВЫБОРКИ ИЗ 9 ИЗОБРАЖЕНИЙ
plt.figure(figsize=(10, 10))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        print(labels)
        plt.title(int(labels[i][1]))
        plt.axis("off")
plt.show()

if TRAINING_MODE:
    # ФОРМИРОВАНИЕ ДАТАСЕТА ДЛЯ ОБУЧЕНИЯ
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_PATH,
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=12418,
        validation_split=0.2,
        subset="training",
        interpolation="bilinear"
    )
    # ФОРМИРОВАНИЕ ДАТАСЕТА ДЛЯ ВАЛИДАЦИИ
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_PATH,
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=12418,
        validation_split=0.2,
        subset="validation",
        interpolation="bilinear"
    )
    # ВЫВОД ВЫБОРКИ ИЗ 9 ИЗОБРАЖЕНИЙ
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(int(labels[i][1]))
            plt.axis("off")
    plt.show()

    # ОБУЧЕНИЕ
    train_ds = train_ds.prefetch(buffer_size=batch_size)
    val_ds = val_ds.prefetch(buffer_size=batch_size)

    # model = make_model(input_shape=image_size + (3,), num_classes=2)
    model = models.vgg(image_size + (3,), n_classes=2)
    keras.utils.plot_model(model, show_shapes=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint("Models/save_at_{epoch}.h5"),
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(acc),
        # tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9, nesterov=True),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    hist = model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
    )
    model.evaluate(test_ds)  # Тестовая выборка

    plt.grid()  # включение отображение сетки
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    axes = plt.gca()
    axes.set_ylim([0, 0.5])
    plt.show()


else:
    model = tf.keras.models.load_model(load_model)

    # model.evaluate(test_ds)  # Тестовая выборка
    # predict = model.predict(true_test_x)
    # for x in range(len(predict)):
    #     if int_r(predict[x]) != 1:
    #         print(predict[x], "-", x)
    #         #print(test_y[x])
    #         print("values:", int_r(predict[x]), "and", test_y[x][0])
    #         cv2.imshow(str(x), true_test_x[x])
    #         cv2.imwrite("G:/PyProjects/OldNet/Data/NegativePositive/" + str(x) + ".jpg", true_test_x[x])

    predict = model.predict(inputImages32)
    filelist = [f for f in os.listdir("G:/PyProjects/OldNet/Data/Output_mser/")]
    for f in filelist:
        os.remove(os.path.join("G:/PyProjects/OldNet/Data/Output_mser/", f))

    for x in range(len(predict)):
        if int_r(predict[x][1]) == 1:
            print(predict[x][1], "-", x)
            print("values:", int_r(predict[x][1]))
            cv2.rectangle(input_image, (tuples[x][0], tuples[x][1]), (tuples[x][2], tuples[x][3]),
                          (0, 100, 255), 1)
            cv2.imwrite("G:/PyProjects/OldNet/Data/Output_mser/" + str(tuples[x]) + ".png", areas[x])
    cv2.imshow("Output", input_image)

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
