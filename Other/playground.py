import os
import cv2
from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


datagen = ImageDataGenerator(
        brightness_range=[0.5, 1.5],
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest')

pathO = "G:/PyProjects/OldNet/Data/New/"
pathI = "G:/PyProjects/OldNet/Data/Old/"

filelist = [f for f in os.listdir(pathO)]
for f in filelist:
    os.remove(os.path.join(pathO, f))


input = [f for f in os.listdir(pathI)]
for f in input:
    img = load_img(pathI+str(f))  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='Data/New', save_prefix=f[:-3], save_format='png'):
        i += 1
        if i > 5:
            break  # otherwise the generator would loop indefinitely

# pathO = "G:/PyProjects/OldNet/Data/12/"
# filelist = [f for f in os.listdir(pathO)]
# file = [cv2.imread(os.path.join(pathO, f)) for f in os.listdir(pathO)]
# x=0
# y=0
# for f in file:
#     if f.shape[0] <65 and  f.shape[1]<65:
#         print(filelist[x], f.shape)
#         os.remove(os.path.join(pathO, filelist[x]))
#         y+=1
#     x+=1
#
# print(y)