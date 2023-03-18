import pandas as pd
import numpy as np
import os
import shutil

import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.optimizers import adam_v2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef


def mcc_metric(y_true, y_pred):
    predicted = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
    true_pos = tf.math.count_nonzero(predicted * y_true)
    true_neg = tf.math.count_nonzero((predicted - 1) * (y_true - 1))
    false_pos = tf.math.count_nonzero(predicted * (y_true - 1))
    false_neg = tf.math.count_nonzero((predicted - 1) * y_true)
    x = tf.cast((true_pos + false_pos) * (true_pos + false_neg)
                * (true_neg + false_pos) * (true_neg + false_neg), tf.float32)
    return tf.cast((true_pos * true_neg) - (false_pos * false_neg), tf.float32) / tf.sqrt(x)


path_f = 'G:\\CCUP_2023\\2023\\'

path_test = 'G:\\CCUP_2023\\Train_Test_Folder\\test'
path_train = 'G:\\CCUP_2023\\Train_Test_Folder\\train'
path_val = 'G:\\CCUP_2023\\Train_Test_Folder\\val'

path_src = 'G:\\CCUP_2023\\2023\\images\\'
path_dst = 'G:\\CCUP_2023\\2023\\sort\\'
path_result_test = 'G:\\CCUP_2023\\2023\\result_test\\'

df = pd.read_csv('G:\\CCUP_2023\\2023\\classes.csv', sep=',', engine='python')

image_data = []
image_labels = []
IMG_HEIGHT = 40
IMG_WIDTH = 40
channels = 3

# РАЗБИВКА начало
# os.mkdir(path_f+'sort\\')
'''
for i in range(0,len(df)):
    os.mkdir(path_f+'sort\\'+str(i))

df_photo = pd.read_csv('G:\\CCUP_2023\\2023\\train.csv',sep=',',engine='python')

from PIL import Image
countErr=0
countPass=0
for index, row in df_photo.iterrows():

    #print(str(index)+' '+row['image'])
    img = row['image']
    try:
        countPass+=1
        Image.open(path_src + img)#ищем плохие картинки и выбрасываем их
        shutil.copyfile(path_src + row['image'], path_dst + str(row['class_id']) + '\\' + row['image'])
    except:
        countErr+=1
        print("----------Error in" + img)

print('  Кол-во ошибок перенос картинок '+str(countErr))
print('  Перенесено успешно '+str(countPass))
print()
print('OK-1')
print()



import python_splitter
python_splitter.split_from_folder(path_f+"sort", train=0.8, test=0.2, val=0)
#РАЗБИВКА конец
'''

'''
#перенос финальных картинок
df_photo = pd.read_csv('G:\\CCUP_2023\\2023\\test.csv',sep=',',engine='python')

from PIL import Image
countErr=0
countPass=0
for index, row in df_photo.iterrows():

    #print(str(index)+' '+row['image'])
    img = row['image']
    try:
        countPass+=1
        #Image.open(path_src + img)#ищем плохие картинки и выбрасываем их
        shutil.copyfile(path_src + row['image'], path_result_test + '\\' + row['image'])
    except:
        countErr+=1
        shutil.copyfile(path_src + row['image'], path_result_test + '\\' + row['image'])
        print("----------Error in" + img)

print('  Кол-во ошибок перенос картинок '+str(countErr))
print('  Перенесено успешно '+str(countPass))
print()
print('OK-11')
print()
#перенос финальных картинок
'''

'''
count = 0
for i in range(NUM_CATEGORIES):
    #path = path_f+'sort-beta'
    path = path_dst
    images = os.listdir(path + '\\' + str(i))

    for img in images:
        try:
            print(str(count))
            count = count + 1
            image = cv2.imread(path + '\\' + str(i) + '\\' + img)
            image_fromarray = Image.fromarray(image, 'RGB')
            resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
            image_data.append(np.array(resize_image))
            image_labels.append(i)
        except:
            print("Error in " + str(i) + ' ' + img)

print()
print('OK-3')
print()
#старый генератор
'''

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout

NUM_CATEGORIES = len(os.listdir(path_dst))

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

img_size = 100 #для первой
#img_size = 128  # для реснета
batch_size = 128

all_datagen = ImageDataGenerator(rescale=1 / 255., width_shift_range=0.2, height_shift_range=0.2,
                                   zoom_range=0.2, fill_mode='nearest')

train_datagen = ImageDataGenerator(rescale=1 / 255.)
test_datagen = ImageDataGenerator(rescale=1 / 255.)

train_generator = train_datagen.flow_from_directory(path_train,
                                                    target_size=(img_size, img_size),
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(path_test,
                                                  target_size=(img_size, img_size),
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  class_mode='categorical')

all_generator = train_datagen.flow_from_directory(path_dst,
                                                 target_size=(img_size, img_size),
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 class_mode='categorical')

from keras import layers, models

shape_img = (img_size, img_size, 3)

#первая модель не может взлететь выше 0
model = models.Sequential(name='123')
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=shape_img))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(256, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.25))
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(NUM_CATEGORIES, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', mcc_metric])



'''
from keras.applications.mobilenet_v2 import MobileNetV2

base_model = MobileNetV2(include_top=False,
                         weights='imagenet',
                         input_shape=(img_size, img_size, 3))


model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))  # to prevent overfitting
model.add(Dense(NUM_CATEGORIES, activation='softmax'))
'''

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', mcc_metric])

model.load_weights('2023-3.h5')

early_stop = tf.keras.callbacks.EarlyStopping(restore_best_weights=True, monitor='loss', patience=100)

checkpoint_1 = tf.keras.callbacks.ModelCheckpoint("2023-3.h5", monitor='loss', verbose=2, save_best_only=True,
                                                  mode='min')

epochs = 10000
history = model.fit(all_generator, epochs=epochs, batch_size=batch_size, validation_data=test_generator, callbacks=[early_stop, checkpoint_1])


# model.save('2023.h5')


'''
#Проверка коефа матью
from sklearn.metrics import matthews_corrcoef
count = 0
y_true = []
X_test = []
for i in range(14):
    #path = path_test
    path = 'G:\\CCUP_2023\\2023\\sort-beta\\'
    images = os.listdir(path + '\\' + str(i))
    count = 0
    for img in images:
        try:
            count+=1
            print(str(count))
            count = count + 1
            image = path + '\\' + str(i) + '\\' + img
            print(image)
            X_test.append([image,i])
            y_true.append(i)
        except:
            print("------Error in " + str(i) + ' ' + img)

print()
print('OK-5')
print()

y_pred = []
from keras_preprocessing import image
promah=0
popal=0
for img1,num in X_test:
    img = image.load_img(img1, target_size=(100, 100))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.
    predictions = model.predict(x)
    max_index = np.argmax(predictions[0])
    y_pred.append(max_index)
    print('True '+str(num)+' pred_max '+str(max_index)+' pred_min '+str(np.argmin(predictions[0])))
    if num==max_index:
        popal+=1
    else:
        promah+=1

mcc = matthews_corrcoef(y_true, y_pred)
print(mcc)
print(' '+str(popal)+' '+str(promah)+' ')
#Проверка коефа матью
'''

'''
df_test = pd.read_csv('G:\\CCUP_2023\\2023\\test.csv', sep=',', engine='python')

result = []
result.append(['class_id', 'image'])
countPass=0
from keras_preprocessing import image
countError = 0
for index, row in df_test.iterrows():
    try:

        countPass+=1
        #image = cv2.imread('G:\\CCUP_2023\\2023\\images\\' + row['image'])
        #image = cv2.resize(image, (img_size, img_size))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = image[np.newaxis, ...]
        #image = image / 255.


        img = image.load_img('G:\\CCUP_2023\\2023\\images\\' + row['image'], target_size=(100, 100))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.

        predict = model.predict(x)
        max_index = np.argmax(predict[0])
        #print(predict[0][max_index])
        print(str(index) + ' ' + str(max_index) + ',' + row['image']+'   '+str(predict[0][max_index]))
        result.append([max_index, row['image']])
    except:
        countError+=1
        print("-----Error in " + str(index) + ' ' + row['image'])
        result.append([0, row['image']])


print('    Проблем обнаружено '+str(countError))
print('    Картинок обработано '+str(countPass))

import pandas as pd
Fcolumns = ['id_class', 'image']

my_df = pd.DataFrame(result, columns=Fcolumns)
my_df.to_csv('my_csv.csv', index=False, header=False)
print('FINAL')
'''

'''
images = []
path = os.path.join(path_result_test, '0a2bTiLScXKsefkYmqWApEuh7yUQJOztxMBFrNG8.jpg')
img = Image.open(path)
rgb_im = img.convert('RGB')
img_for_error = rgb_im.resize((img_size, img_size))

for filename in os.listdir(path_result_test):
    try:
        path = os.path.join(path_result_test, filename)
        img = Image.open(path)
        img = img.resize((img_size, img_size))
        rgb_im = img.convert('RGB')
        images.append(rgb_im)
    except:
        print('Ошибка в '+filename)
        images.append(img_for_error)#забить место чем-то без ошибки

print('началось распознование')
images = np.array([np.array(img) for img in images])

np.save('final_predict.npy', images)    # .npy extension is added if not given
'''
images = np.load('final_predict.npy')

count=0
predictions = []
for i in images:
    x = np.expand_dims(i, axis=0)
    x = x / 255.
    predictions.append(model.predict(x))
    count+=1
    print(str(count))

print('Окончилось распознование')
result = []
result.append(['class_id','image'])

df_test = pd.read_csv('G:\\CCUP_2023\\2023\\test.csv', sep=',', engine='python')

count=0
for i in range(len(images)):
    predicted_class = np.argmax(predictions[i])
    result.append([predicted_class, df_test.at[i, 'image']])
    count+=1
    print(str(count))

my_df = pd.DataFrame(result, columns=['class_id','image'])
my_df.to_csv('my_csv.csv', index=False, header=False)


'''
y_test = test_generator.classes
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)

from sklearn.metrics import classification_report

labels = []

print(classification_report(y_test, y_pred, target_names=labels))
'''