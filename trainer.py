import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import tqdm
import random
from keras.preprocessing.image import load_img
import PIL
import cv2
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf 


from tensorflow.keras.layers import  Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings('ignore')


input_path = []
label = []

for cls in os.listdir("Data"):
    print("Found ", cls)
    for path in os.listdir(os.path.join("Data", cls)):
        print(cls)
        if cls == 'HeatStress':
            label.append(0)
        else:
            label.append(1)
        input_path.append(os.path.join("Data", cls, path))
    print(input_path[0], label[0])

print("LABELS -> ", len(label))
print("LABELS Paths-> ",len(input_path))

df = pd.DataFrame()
df['images'] = input_path
df['label'] = label
df = df.sample(frac = 1).reset_index(drop=True)
df.head()

for i in df['images']:
    if '.jpg' not in i:
        print(i)

len(df)


l = []

for image in df['images']:
    try:
        img = PIL.Image.open(image)
    except Exception as e:
        l.append(image)


plt.figure(figsize=(25,25))
temp = df[df['label']==1]['images']
start = random.randint(0, len(temp))
files = temp[start:start+25]

for index, file in enumerate(files):
    plt.subplot(5,5, index+1)
    img = cv2.imread(file)
    img = np.array(img)
    plt.imshow(img)
    plt.title('HeatStress')
    plt.axis('off')
    plt.savefig('heatstressplot.png')

plt.figure(figsize=(25,25))
temp = df[df['label']==0]['images']
start = random.randint(0, len(temp))
files = temp[start:start+25]

for index, file in enumerate(files):
    plt.subplot(5,5, index+1)
    img = cv2.imread(file)
    img = np.array(img)
    plt.imshow(img)
    plt.title('Normal')
    plt.savefig('normalplot.png')
  

sns.countplot(df['label'])

df['label'] = df['label'].astype('str')

train, test = train_test_split(df, test_size=0.2, random_state=42)

train.head()


train_generator=ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True,
    fill_mode='nearest'
)


val_generator = ImageDataGenerator(
    rescale=1./255
)

train_iterator = train_generator.flow_from_dataframe(
    train, 
    x_col='images', 
    y_col='label', 
    target_size=(120,120), 
    batch_size=512, 
    class_mode='binary'
    )

val_iterator = train_generator.flow_from_dataframe(
    test, 
    x_col='images', 
    y_col='label', 
    target_size=(120,120), 
    batch_size=512, 
    class_mode='binary'
    )


def mai_Net():
    model = tf.keras.models.Sequential([
                                        Conv2D(16, (3,3), activation='relu', input_shape=(120,120,3)),
                                        MaxPool2D((2,2)),
                                        Conv2D(32, (2,2), activation='relu'),
                                        MaxPool2D((2,2)),
                                        Conv2D(64,(3,3), activation='relu'),
                                        MaxPool2D((2,2)),
                                        Flatten(),
                                        Dense(512, activation='relu'),
                                        Dense(1, activation='sigmoid')
    ])

    return model

model = mai_Net()


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(train_iterator, epochs=10, validation_data=val_iterator)

model.save('new.h5')
