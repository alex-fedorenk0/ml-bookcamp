import numpy as np

import os, shutil
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_image_helper import create_preprocessor

# Splitting test folder into val and test
for root, dirs, files in os.walk('./is that santa/'):
    if len(files) > 0:
        print(f'{root}: {len(files)} files')

val_santa = random.sample(os.listdir('./is that santa/test/santa'), 154)
val_not_santa = random.sample(os.listdir('./is that santa/test/not-a-santa'), 154)

test_path = os.path.join(os.getcwd(), 'is that santa/test/')
val_path = os.path.join(os.getcwd(), 'is that santa/val/')

for filename in val_santa:
    shutil.move(
        os.path.join(test_path, 'santa', filename),
        os.path.join(val_path, 'santa', filename)
        )
for filename in val_not_santa:
    shutil.move(
        os.path.join(test_path, 'not-a-santa', filename),
        os.path.join(val_path, 'not-a-santa', filename)
        )

for root, dirs, files in os.walk('./is that santa/'):
    if len(files) > 0:
        print(f'{root}: {len(files)} files')

# Image sets generation and preprocessing

train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_ds = train_gen.flow_from_directory(
    './is that santa/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = val_gen.flow_from_directory(
    './is that santa/val',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_ds = test_gen.flow_from_directory(
    './is that santa/test',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

def make_model(learning_rate=0.001, size_inner=50, drop_rate=0.7):
    base_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(150, 150, 3))
    base = base_model(inputs, training=False)
    pooling = keras.layers.GlobalAveragePooling2D()
    vectors = pooling(base)
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(drop_rate)(inner)
    outputs = keras.layers.Dense(1, activation='sigmoid')(drop)
    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.BinaryCrossentropy()

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model

# Final model training and saving
model = make_model(learning_rate=0.001, size_inner=50, drop_rate=0.7)
history = model.fit(train_ds, epochs=20, validation_data=val_ds)

model.save('santa-class-v1.h5')
tf.saved_model.save(model, 'santa-class-v1')



