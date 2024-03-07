#%% convolutional layers
from sklearn.datasets import load_sample_images
import tensorflow.keras as tfk

# implementing convolutional layers with keras
images = load_sample_images()["images"]
images = tfk.layers.CenterCrop(height=70, width=120)(images)
images = tfk.layers.Rescaling(scale = 1/255)(images)
print(images.shape)

conv_layer = tfk.layers.Conv2D(filters=32, kernel_size=7)
fmaps = conv_layer(images)
print(fmaps.shape)

conv_layer = tfk.layers.Conv2D(filters=32, kernel_size=7, padding="same")
fmaps = conv_layer(images)
print(fmaps.shape)

kernels, biases = conv_layer.get_weights()
print(kernels.shape)
print(biases.shape)

#%% implementing pooling layers with keras
import tensorflow as tf

max_pool = tfk.layers.MaxPool2D(pool_size=2)

class DepthPool(tfk.layers.Layer):
    def __init__(self, pool_size=2, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        
    def call(self, inputs):
        shape = tf.shape(inputs)
        groups = shape[-1] //self.pool_size
        new_shape = tf.concat([shape[:-1], [groups, self.pool_size]], axis=0)
        return tf.reduce_max(tf.reshape(inputs, new_shape), axis=-1)

global_avg_pool = tfk.layers.GlobalAvgPool2D()
global_avg_pool = tfk.layers.Lambda(lambda X: tf.reduce_mean(X, axis=[1, 2]))
print(global_avg_pool(images))

#%% CNN architectures
from functools import partial

DefaultConv2D = partial(tfk.layers.Conv2D, kernel_size=3, padding="same",
                        activation="relu", kernel_initializer="he_normal")
model = tfk.Sequential([
    DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
    tfk.layers.MaxPool2D(),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    tfk.layers.MaxPool2D(),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    tfk.layers.MaxPool2D(),
    tfk.layers.Flatten(),
    tfk.layers.Dense(units=128, activation="relu", kernel_initializer="he_normal"),
    tfk.layers.Dropout(0.5),
    tfk.layers.Dense(units=64, activation="relu", kernel_initializer="he_normal"),
    tfk.layers.Dropout(0.5),
    tfk.layers.Dense(units=10, activation="softmax")
    ])

#%% implementing a resnet-34 CNN using keras

DefaultConv2D = partial(tfk.layers.Conv2D, kernel_size=3, strides=1, padding="same",
                        kernel_initializer="he_normal", use_bias=False)

class ResidualUnit(tfk.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tfk.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            tfk.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            tfk.layers.BatchNormalization()
            ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                tfk.layers.BatchNormalization()
                ]
        
    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)
    
model = tfk.Sequential([
    DefaultConv2D(64, kernel_size=7, strides=2, input_shape=[224, 224, 3]),
    tfk.layers.BatchNormalization(),
    tfk.layers.Activation("relu"),
    tfk.layers.MaxPool2D(pool_size=3, strides=2, padding="same")
    ])
prev_filters=64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters

model.add(tfk.layers.GlobalAvgPool2D())
model.add(tfk.layers.Flatten())
model.add(tfk.layers.Dense(10, activation="softmax"))

#%% using pre-trained models from keras

model = tfk.applications.ResNet50(weights="imagenet")

images = load_sample_images()["images"]
images_resized = tfk.layers.Resizing(height=224, width=224,
                                     crop_to_aspect_ratio=True)(images)

inputs = tfk.applications.resnet50.preprocess_input(images_resized)
y_proba = model.predict(inputs)
print(y_proba.shape)

top_k = tfk.applications.resnet50.decode_predictions(y_proba, top=3)
for image_idx in range(len(images)):
    print(f"Image #{image_idx}")
    for class_id, name, y_proba in top_k[image_idx]:
        print(f"    {class_id} - {name:12s} {y_proba:.2%}")

#%% pre-trained models for transfer learning
# this section requires a GPU (or training is VERY slow)
import tensorflow_datasets as tfds

dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
dataset_size = info.splits["train"].num_examples
print(dataset_size)
class_names = info.features["label"].names
print(class_names)
n_classes = info.features["label"].num_classes
print(n_classes)

test_set_raw, val_set_raw, train_set_raw = tfds.load(
    "tf_flowers",
    split=["train[:10%]", "train[10%:25%]", "train[25%:]"],
    as_supervised=True)

batch_size = 32
preprocess = tfk.Sequential([
    tfk.layers.Resizing(height=224, width=224, crop_to_aspect_ratio=True),
    tfk.layers.Lambda(tfk.applications.xception.preprocess_input)
    ])
train_set = train_set_raw.map(lambda X, y: (preprocess(X), y))
train_set = train_set.shuffle(1000, seed=42).batch(batch_size).prefetch(1)
val_set = val_set_raw.map(lambda X, y: (preprocess(X), y)).batch(batch_size)
test_set = test_set_raw.map(lambda X, y: (preprocess(X), y)).batch(batch_size)

data_augmentation = tfk.Sequential([
    tfk.layers.RandomFlip(mode="horizontal", seed=42),
    tfk.layers.RandomRotation(factor=0.05, seed=42),
    tfk.layers.RandomContrast(factor=0.2, seed=42)
    ])

base_model = tfk.applications.xception.Xception(weights="imagenet", include_top=False)
avg = tfk.layers.GlobalAveragePooling2D()(base_model.output)
output = tfk.layers.Dense(n_classes, activation="softmax")(avg)
model = tfk.Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

optimiser = tfk.optimizers.SGD(learning_rate=0.1, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimiser,
              metrics=["accuracy"])
history = model.fit(train_set, validation_data=val_set, epochs=3)

for layer in base_model.layers[56:]:
    layer.trainable = True

optimiser = tfk.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimiser,
              metrics=["accuracy"])
history = model.fit(train_set, validation_data=val_set, epochs=10)

#%% classification and localisation

base_model = tfk.applications.xception.Xception(weights="imagenet", include_top=False)
avg = tfk.layers.GlobalAveragePooling2D()(base_model.output)
class_output = tfk.layers.Dense(n_classes, activation="softmax")(avg)
loc_output = tfk.layers.Dense(4)(avg)
model = tfk.Model(inputs=base_model.input, outputs=[class_output, loc_output])
model.compile(loss=["sparse_categorical_crossentropy", "mse"], loss_weights=[0.8, 0.2],
              optimizer=optimiser, metrics=["accuracy"])

#%% Coding Exercises: Exercise 9
# this section requires a GPU (or training is VERY slow)

import tensorflow.keras as tfk
import numpy as np
import tensorflow as tf

# build a CNN from scratch and try and achieve the highest possible accuracy on MNIST
(X_train_val, y_train_val), (X_test, y_test) = tfk.datasets.mnist.load_data()
X_train_val = X_train_val / 255.
X_train, X_val = X_train_val[:-5000], X_train_val[-5000:]
y_train, y_val = y_train_val[:-5000], y_train_val[-5000:]
X_test = X_test / 255.

X_train = X_train.reshape(-1, 28, 28, 1)
X_val = X_val.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

tfk.backend.clear_session()
tf.random.set_seed(42)

data_augmentation = tfk.Sequential([
    tfk.layers.RandomFlip(mode="horizontal", seed=42),
    tfk.layers.RandomRotation(factor=0.05, seed=42),
    tfk.layers.RandomContrast(factor=0.2, seed=42)
    ])

model = tfk.models.Sequential()
model.add(data_augmentation)
for _ in range(3):
    model.add(tfk.layers.Conv2D(filters=64, kernel_size=3, padding="same",
                                activation="swish", kernel_initializer="he_normal"))
model.add(tfk.layers.MaxPool2D())
for _ in range(2):
    model.add(tfk.layers.Conv2D(filters=128, kernel_size=3, padding="same",
                                activation="swish", kernel_initializer="he_normal"))
model.add(tfk.layers.MaxPool2D())
model.add(tfk.layers.Conv2D(filters=256, kernel_size=3, padding="same",
                            activation="swish", kernel_initializer="he_normal"))
model.add(tfk.layers.Flatten())
model.add(tfk.layers.Dropout(0.25))
model.add(tfk.layers.Dense(256, activation="swish", kernel_initializer="he_normal"))
model.add(tfk.layers.Dropout(0.5))
model.add(tfk.layers.Dense(10, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])

reduce_lr_cb = tfk.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, verbose=1,
                                               patience=2, min_lr=1e-8)
model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val),
          callbacks=[reduce_lr_cb])

model.summary()

model.evaluate(X_val, y_val)

#%% Coding Exercises: Exercise 10
# this section requires a GPU (or training is VERY slow)
import os
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tensorflow.train import Example, Features, Feature, BytesList, Int64List
from tensorflow.data import TFRecordDataset
import matplotlib.pyplot as plt
import tensorflow.keras as tfk
import numpy as np
import pandas as pd

# create a training set containing at least 100 images per class. You can classify your
# own pictures or use a dataset from tensorflow datasets

# the images are labelled by the file prefixes:
    # 01 = Akhal-Teke
    # 02 = Appaloosa
    # 03 = Orlov Trotter
    # 04 = Vladimir Heavy Draft
    # 05 = Percheron
    # 06 = Arabian
    # 07 = Friesian

source_dir = "./datasets/horse_breeds"
train_dir = os.path.join(source_dir, "train")
val_dir = os.path.join(source_dir, "val")
test_dir = os.path.join(source_dir, "test")

for data_dir in [train_dir, val_dir, test_dir]:
    os.makedirs(data_dir, exist_ok=True)

prefix_dict = {}
for file in os.listdir(source_dir):
    if file.endswith(".png"):
        prefix = file[:2]
        if prefix not in prefix_dict:
            prefix_dict[prefix] = []
        prefix_dict[prefix].append(file)
prefix_dict = {prefix: files for prefix, files in prefix_dict.items() if len(files) >= 100}
    
# split it into a training, validation and test set
for prefix, files in prefix_dict.items():
    train_set, val_test_set = train_test_split(files, test_size=0.2, random_state=42)
    val_set, test_set = train_test_split(val_test_set, test_size=0.5, random_state=42)
    
    for file in train_set:
        source_path = os.path.join(source_dir, file)
        target_path = os.path.join(train_dir, file)
        shutil.move(source_path, target_path)
    
    for file in val_set:
        source_path = os.path.join(source_dir, file)
        target_path = os.path.join(val_dir, file)
        shutil.move(source_path, target_path)
        
    for file in test_set:
        source_path = os.path.join(source_dir, file)
        target_path = os.path.join(test_dir, file)
        shutil.move(source_path, target_path)


def load_and_process_image(image_path):
    img = load_img(image_path)
    img_array = img_to_array(img)
    img_array = tf.image.convert_image_dtype(img_array, dtype=tf.float32)
    return img_array

# def create_tensor_dataset(folder_path, target_size=(224, 224)):
#     images = []
#     labels = []

#     for filename in os.listdir(folder_path):
#         if filename.endswith('.png'):
#             image_path = os.path.join(folder_path, filename)
#             img_array = load_and_process_image(image_path)           
#             img_array = tf.image.resize(img_array, target_size)
#             images.append(img_array)

#             label = int(filename[:2])
#             labels.append(label)

#     images = tf.convert_to_tensor(images)
#     labels = tf.convert_to_tensor(labels)

#     dataset = tf.data.Dataset.from_tensor_slices((images, labels))
#     return dataset
        
# train_dataset = create_tensor_dataset(train_dir)
# val_dataset = create_tensor_dataset(val_dir)
# test_dataset = create_tensor_dataset(test_dir)

def create_tfrecord_file(dataset_dir, source_dir, output_file):
    writer = tf.io.TFRecordWriter(os.path.join(source_dir, output_file))

    for filename in os.listdir(dataset_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(dataset_dir, filename)
            img_tensor = load_and_process_image(img_path)
            img_array = img_tensor.numpy()
            img_encoded = tf.io.encode_jpeg(img_array).numpy()
    
            label = int(filename[:2]) - 1
            
            example = Example(
                features=Features(
                    feature={
                        'image': Feature(bytes_list=BytesList(value=[img_encoded])),
                        'label': Feature(int64_list=Int64List(value=[label]))
            }))

            writer.write(example.SerializeToString())
    writer.close()

create_tfrecord_file(train_dir, source_dir, 'train.tfrecord')
create_tfrecord_file(val_dir, source_dir, 'val.tfrecord')
create_tfrecord_file(test_dir, source_dir, 'test.tfrecord')

# build the input pipeline, apply the appropriate preprocessing operations and
# optionally add data augmentation
root_dir = "./datasets/horse_breeds"
class_names = ["Akhal-Teke", "Appalossa", "Orlov Trotter", "Vladimir Heavy Draft",
               "Percheron", "Arabian", "Friesian"]

def parse_tfrecord(tfrecord):
    feature_description = {
        "image": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "label": tf.io.FixedLenFeature(shape=[], dtype=tf.int64)
        }
    example = tf.io.parse_single_example(tfrecord, feature_description)
    image = tf.io.decode_jpeg(example["image"], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    label = example["label"]
    
    return image, label


def horse_breed_dataset(filepath, batch_size=32, cache=True, shuffle=True):
    dataset = TFRecordDataset(filepath)
    if cache:
        dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, seed=42)
    dataset = dataset.map(parse_tfrecord)
    # dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)

train_set_raw = horse_breed_dataset(os.path.join(root_dir, 'train.tfrecord'))
val_set_raw = horse_breed_dataset(os.path.join(root_dir, 'val.tfrecord'), shuffle=True)
test_set_raw = horse_breed_dataset(os.path.join(root_dir, 'test.tfrecord'), shuffle=False)

idx = 0
for img, label in val_set_raw.take(9):
    idx += 1
    plt.subplot(3, 3, idx)
    plt.imshow(img)
    plt.title(f"Breed: {class_names[label]}")
    plt.axis("off")
    plt.tight_layout()

tfk.backend.clear_session()
batch_size = 32
preprocess = tfk.models.Sequential([
    tfk.layers.Resizing(height=224, width=224, crop_to_aspect_ratio=True),
    tfk.layers.Lambda(tfk.applications.efficientnet_v2.preprocess_input)
    ])
train_set = train_set_raw.map(lambda X, y: (preprocess(X), y)).batch(batch_size)
val_set = val_set_raw.map(lambda X, y: (preprocess(X), y)).batch(batch_size)
test_set = test_set_raw.map(lambda X, y: (preprocess(X), y)).batch(batch_size)

for X_batch, y_batch in val_set.take(1):
    for idx in range(9):
        plt.subplot(3, 3, idx+1)
        plt.imshow(X_batch[idx])
        plt.title(f"Breed: {class_names[y_batch[idx]]}")
        plt.axis("off")
        plt.tight_layout()
        
data_augmentation = tfk.Sequential([
    tfk.layers.RandomFlip(mode="horizontal", seed=42),
    tfk.layers.RandomContrast(factor=0.2, seed=42),
    tfk.layers.RandomBrightness(factor=0.2, seed=42)
    ])

for X_batch, y_batch in val_set.take(1):
    X_batch_augmented = data_augmentation(X_batch, training=True)
    for idx in range(9):
        plt.subplot(3, 3, idx+1)
        plt.imshow(np.clip(X_batch[idx], 0, 1))
        plt.title(f"Breed: {class_names[y_batch[idx]]}")
        plt.axis("off")
        plt.tight_layout()

# fine-tune a pretrained model on this dataset
# for _, y_batch in train_set:
#     num_classes = len(np.unique(y_batch))
num_classes = len(class_names)

tfk.backend.clear_session()
tf.random.set_seed(42)
base_model = tfk.applications.efficientnet_v2.EfficientNetV2S(
    weights="imagenet", include_top=False)
avg = tfk.layers.GlobalAveragePooling2D()(base_model.output)
output = tfk.layers.Dense(num_classes, activation="softmax")(avg)
model = tfk.Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False
    
optimiser = tfk.optimizers.Nadam(learning_rate=0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimiser,
              metrics=["accuracy"])
history = model.fit(train_set, validation_data=val_set, epochs=20)

def plot_fit_history(fit_history):
    history_df = pd.DataFrame(fit_history.history)
    fig, axs = plt.subplots(2, 1, sharex=True)
    history_df[["loss", "val_loss"]].plot(ax=axs[0], style=["b-", "g--."])
    history_df[["accuracy", "val_accuracy"]].plot(ax=axs[1], style=["b-", "g--."])
    
plot_fit_history(history)

for layer in base_model.layers[-101:]:
    layer.trainable = True

optimiser = tfk.optimizers.Nadam(learning_rate=0.001)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimiser,
              metrics=["accuracy"])

reduce_lr_cb = tfk.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, verbose=1,
                                               patience=2, min_lr=1e-8)
early_stopping_cb = tfk.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                                restore_best_weights=True)
history = model.fit(train_set, validation_data=val_set, epochs=30,
                    callbacks=[reduce_lr_cb, early_stopping_cb])

plot_fit_history(history)
