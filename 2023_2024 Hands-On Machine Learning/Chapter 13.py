#%% the tf.data API
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import tensorflow.keras as tfk

X = tf.range(10)
dataset = tf.data.Dataset.from_tensor_slices(X)
print(dataset)

for item in dataset:
    print(item)
    
X_nested = {"a": ([1, 2, 3], [4, 5, 6]), "b": [7, 8, 9]}
dataset = tf.data.Dataset.from_tensor_slices(X_nested)
for item in dataset:
    print(item)

# chaining transformations
dataset = tf.data.Dataset.from_tensor_slices(tf.range(10))
dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print(item)

dataset = dataset.map(lambda x: x**2)
for item in dataset:
    print(item)

dataset = dataset.filter(lambda x: tf.reduce_sum(x) > 50)
for item in dataset:
    print(item)

# shuffling the data
dataset = tf.data.Dataset.range(10).repeat(2)
dataset = dataset.shuffle(buffer_size=4, seed=42).batch(7)
for item in dataset:
    print(item)
    
# interleaving lines from multiple files
housing = fetch_california_housing()
X_train_val, X_test, y_train_val, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, random_state=42)

def save_to_csv_files(data, name_prefix, header=None, n_parts=10):
    housing_dir = Path() / "datasets" / "housing"
    housing_dir.mkdir(parents=True, exist_ok=True)
    filename_format = "my_{}_{:02d}.csv"

    filepaths = []
    m = len(data)
    chunks = np.array_split(np.arange(m), n_parts)
    for file_idx, row_indices in enumerate(chunks):
        part_csv = housing_dir / filename_format.format(name_prefix, file_idx)
        filepaths.append(str(part_csv))
        with open(part_csv, "w") as f:
            if header is not None:
                f.write(header)
                f.write("\n")
            for row_idx in row_indices:
                f.write(",".join([repr(col) for col in data[row_idx]]))
                f.write("\n")
    return filepaths

train_data = np.c_[X_train, y_train]
val_data = np.c_[X_val, y_val]
test_data = np.c_[X_test, y_test]
header_cols = housing.feature_names + ["MedianHouseValue"]
header = ",".join(header_cols)

train_filepaths = save_to_csv_files(train_data, "train", header, n_parts=20)
val_filepaths = save_to_csv_files(val_data, "val", header, n_parts=10)
test_filepaths = save_to_csv_files(test_data, "test", header, n_parts=10)

train_filepaths

filepath_dataset = tf.data.Dataset.list_files(train_filepaths, seed=42)

n_readers = 5
dataset = filepath_dataset.interleave(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1), cycle_length=n_readers)

for line in dataset.take(5):
    print(line.numpy())

# preprocessing the data
scaler = StandardScaler()
scaler.fit(X_train)

X_mean, X_std = scaler.mean_, scaler.scale_
n_inputs = 8

def parse_csv_line(line):
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    return tf.stack(fields[:-1]), tf.stack(fields[-1:])

def preprocess(line):
    x, y = parse_csv_line(line)
    return (x - X_mean) / X_std, y

preprocess(b'4.2083,44.0,5.3232,0.9171,846.0,2.3370,37.47,-122.2,2.782')

# putting everything together
def csv_reader_dataset(filepaths, n_readers=5, n_read_threads=None, n_parse_threads=5,
                       shuffle_buffer_size=10000, seed=42, batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths, seed=seed)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, num_parallel_calls=n_read_threads)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)
    return dataset.batch(batch_size).prefetch(1)

# using the dataset with keras
train_set = csv_reader_dataset(train_filepaths)
val_set = csv_reader_dataset(val_filepaths)
test_set = csv_reader_dataset(test_filepaths)

model = tfk.Sequential([
    tfk.layers.Dense(30, activation='relu', kernel_initializer='he_normal',
                     input_shape=X_train.shape[1:]),
    tfk.layers.Dense(1)
    ])
model.compile(loss='mse', optimizer='sgd')
model.fit(train_set, validation_data=val_set, epochs=5)

test_mse = model.evaluate(test_set)
print(test_mse)
new_set = test_set.take(3)
y_pred = model.predict(new_set)

optimiser = tfk.optimizers.SGD(learning_rate=0.01)
loss_fn = tfk.losses.mean_squared_error
n_epochs = 5
for epoch in range(n_epochs):
    for X_batch, y_batch in train_set:
        print("\rEpoch {}/{}".format(epoch+1, n_epochs), end="")
        with tf.GradientTape() as tape:
            y_pred = model(X_batch)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimiser.apply_gradients(zip(gradients, model.trainable_variables))

@tf.function
def train_one_epoch(model, optimiser, loss_fn, train_set):
    for X_batch, y_batch in train_set:
        with tf.GradientTape() as tape:
            y_pred = model(X_batch)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimiser.apply_gradients(zip(gradients, model.trainable_variables))
optimiser = tfk.optimizers.SGD(learning_rate=0.01)
loss_fn = tfk.losses.mean_squared_error
for epoch in range(n_epochs):
    print("\rEpoch {}/{}".format(epoch+1, n_epochs), end="")
    train_one_epoch(model, optimiser, loss_fn, train_set)

#%% the TFRecord format
import tensorflow as tf
from tensorflow.train import (BytesList, FloatList, Int64List, Feature, Features, Example,
                              FeatureList, FeatureLists, SequenceExample)

with tf.io.TFRecordWriter("./datasets/my_data.tfrecord") as f:
    f.write(b"This is the first record")
    f.write(b"And this is the second record")

filepaths = ["./datasets/my_data.tfrecord"]
dataset = tf.data.TFRecordDataset(filepaths)
for item in dataset:
    print(item)
    
# compressed TFRecord files
options = tf.io.TFRecordOptions(compression_type="GZIP")
with tf.io.TFRecordWriter("./datasets/my_compressed.tfrecord", options) as f:
    f.write(b"Compress, compress, compress!")
    
dataset = tf.data.TFRecordDataset(["./datasets/my_compressed.tfrecord"],
                                  compression_type="GZIP")

# a brief introduction to protocol buffers
%%writefile ./datasets/person.proto
syntax = "proto3"
message Person {
    string name = 1;
    int32 id = 2;
    repeated string email = 3;
    }

# tensorflow protobuffs
person_example = Example(
    features=Features(
        feature={
            "name": Feature(bytes_list=BytesList(value=[b"Alice"])),
            "id": Feature(int64_list=Int64List(value=[123])),
            "emails": Feature(bytes_list=BytesList(value=[b"a@b.com",
                                                          b"c@d.com"]))
            }))

with tf.io.TFRecordWriter("./datasets/my_contacts.tfrecord") as f:
    for _ in range(5):
        f.write(person_example.SerializeToString())

# loading and parsing examples
feature_description = {
    "name": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "emails": tf.io.VarLenFeature(tf.string)
    }

def parse(serialised_example):
    return tf.io.parse_single_example(serialised_example, feature_description)

dataset = tf.data.TFRecordDataset(["./datasets/my_contacts.tfrecord"]).map(parse)
for parsed_example in dataset:
    print(parsed_example)
    
print(tf.sparse.to_dense(parsed_example["emails"], default_value=b""))
print(parsed_example["emails"].values)

def parse(serialised_example):
    return tf.io.parse_example(serialised_examples, feature_description)

dataset = tf.data.TFRecordDataset(["./datasets/my_contacts.tfrecord"]).batch(2)
for parsed_examples in dataset:
    print(parsed_examples)

# handling lists of lists using the SequenceExample protobuff
context = Features(feature={
    "author_id": Feature(int64_list=Int64List(value=[123])),
    "title": Feature(bytes_list=BytesList(value=[b"A", b"desert", b"place", b"."])),
    "pub_date": Feature(int64_list=Int64List(value=[1623, 12, 25]))
})

content = [["When", "shall", "we", "three", "meet", "again", "?"],
           ["In", "thunder", ",", "lightning", ",", "or", "in", "rain", "?"]]
comments = [["When", "the", "hurlyburly", "'s", "done", "."],
            ["When", "the", "battle", "'s", "lost", "and", "won", "."]]

def words_to_feature(words):
    return Feature(bytes_list=BytesList(value=[word.encode("utf-8") for word in words]))

content_features = [words_to_feature(sentence) for sentence in content]
comments_features = [words_to_feature(comment) for comment in comments]
            
sequence_example = SequenceExample(
    context=context,
    feature_lists=FeatureLists(feature_list={
        "content": FeatureList(feature=content_features),
        "comments": FeatureList(feature=comments_features)
    }))

serialised_sequence_example = sequence_example.SerializeToString()

context_feature_descriptions = {
    "author_id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "title": tf.io.VarLenFeature(tf.string),
    "pub_date": tf.io.FixedLenFeature([3], tf.int64, default_value=[0, 0, 0]),
}
sequence_feature_descriptions = {
    "content": tf.io.VarLenFeature(tf.string),
    "comments": tf.io.VarLenFeature(tf.string),
}

parsed_context, parsed_feature_lists = tf.io.parse_single_sequence_example(
    serialised_sequence_example, context_feature_descriptions,
    sequence_feature_descriptions)
parsed_content = tf.RaggedTensor.from_sparse(parsed_feature_lists["content"])
print(parsed_content)

#%% keras preprocessing layers
import pandas as pd
import tensorflow_hub as hub
from sklearn.datasets import load_sample_images

# the normalisation layer
tfk.backend.clear_session()
norm_layer = tfk.layers.Normalization()
model = tfk.models.Sequential([
    norm_layer,
    tfk.layers.Dense(1)
    ])
model.compile(loss="mse", optimizer=tfk.optimizers.SGD(learning_rate=2e-3))
norm_layer.adapt(X_train)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5)

norm_layer = tfk.layers.Normalization()
norm_layer.adapt(X_train)
X_train_scaled = norm_layer(X_train)
X_val_scaled = norm_layer(X_val)

tfk.backend.clear_session()
model = tfk.models.Sequential([tfk.layers.Dense(1)])
model.compile(loss="mse", optimizer=tfk.optimizers.SGD(learning_rate=2e-3))
model.fit(X_train_scaled, y_train, epochs=5, validation_data=(X_val_scaled, y_val))

tfk.backend.clear_session()
final_model = tfk.Sequential([norm_layer, model])
X_new = X_test[:3]
y_pred = final_model(X_new)

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(5)
dataset = dataset.map(lambda X, y: (norm_layer(X), y))

class MyNormalisation(tfk.layers.Layer):
    def adapt(self, X):
        self.mean_ = np.mean(X, axis=0, keepdims=True)
        self.std_ = np.std(X, axis=0, keepdims=True)
        
    def call(self, inputs):
        eps = tfk.backend.epsilon()
        return (inputs - self.mean_) / (self.std_ + eps)

# the discretisation layer
age = tf.constant([[10.], [93.], [57.], [18.], [37.], [5.]])
discretise_layer = tfk.layers.Discretization(bin_boundaries=[18., 50.])
categories = discretise_layer(age)
print(categories)

discretise_layer = tfk.layers.Discretization(num_bins=3)
discretise_layer.adapt(age)
age_categories = discretise_layer(age)
print(age_categories)

# the category encoding layer
onehot_layer = tfk.layers.CategoryEncoding(num_tokens=3)
print(onehot_layer(age_categories))

two_age_categories = np.array([[1, 0], [2, 2], [2, 0]])
print(onehot_layer(two_age_categories))

onehot_layer = tfk.layers.CategoryEncoding(num_tokens=3+3)
print(onehot_layer(two_age_categories + [0, 3]))

# the stringlookup layer
cities = ["Auckland", "Paris", "Paris", "San Francisco"]
str_lookup_layer = tfk.layers.StringLookup()
str_lookup_layer.adapt(cities)
print(str_lookup_layer([["Paris"], ["Auckland"], ["Auckland"], ["Montreal"]]))

str_lookup_layer = tfk.layers.StringLookup(output_mode="one_hot")
str_lookup_layer.adapt(cities)
print(str_lookup_layer([["Paris"], ["Auckland"], ["Auckland"], ["Montreal"]]))

str_lookup_layer = tfk.layers.StringLookup(num_oov_indices=5)
str_lookup_layer.adapt(cities)
print(str_lookup_layer([["Paris"], ["Auckland"], ["Foo"], ["Bar"], ["Baz"]]))

# the hashing layer
hashing_layer = tfk.layers.Hashing(num_bins=10)
print(hashing_layer([["Paris"], ["Tokyo"], ["Auckland"], ["Montreal"]]))

# encoding categorical features using embeddings
tf.random.set_seed(42)
embedding_layer = tfk.layers.Embedding(input_dim=5, output_dim=2)
print(embedding_layer(np.array([2, 4, 2])))

tf.random.set_seed(42)
ocean_prox = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
str_lookup_layer = tfk.layers.StringLookup()
str_lookup_layer.adapt(ocean_prox)

lookup_and_embed = tfk.Sequential([
    tfk.layers.InputLayer(input_shape=[], dtype=tf.string),
    str_lookup_layer,
    tfk.layers.Embedding(input_dim=str_lookup_layer.vocabulary_size(), output_dim=2),
    ])
print(lookup_and_embed(np.array(["<1H OCEAN", "ISLAND", "<1H OCEAN"])))

housing = pd.read_csv('./datasets/housing.csv')
X = housing.drop('median_house_value', axis=1)
y = housing.median_house_value
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, random_state=42)

X_train_num = np.array(X_train.drop('ocean_proximity', axis=1))
X_train_cat = np.array(X_train.ocean_proximity)
X_val_num = np.array(X_val.drop('ocean_proximity', axis=1))
X_val_cat = np.array(X_val.ocean_proximity)

scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train_num)
X_val_num = scaler.transform(X_val_num)

tfk.backend.clear_session()
num_input = tfk.layers.Input(shape=[8], name="num")
cat_input = tfk.layers.Input(shape=[], dtype=tf.string, name="cat")
cat_embeddings = lookup_and_embed(cat_input) 
encoded_inputs = tfk.layers.concatenate([num_input, cat_embeddings])
outputs = tfk.layers.Dense(1)(encoded_inputs)
model = tfk.models.Model(inputs=[num_input, cat_input], outputs=[outputs])
model.compile(loss="mse", optimizer="sgd")
history = model.fit((X_train_num, X_train_cat), y_train, epochs=5,
                    validation_data=((X_val_num, X_val_cat), y_val))

# text pre-processing
train_data = ["To be", "!(to be)", "That's the question", "Be, be, be."]
text_vec_layer = tfk.layers.TextVectorization()
text_vec_layer.adapt(train_data)
print(text_vec_layer(["Be good!", "Question: be or be?"]))

text_vec_layer = tfk.layers.TextVectorization(output_mode="tf_idf")
text_vec_layer.adapt(train_data)
print(text_vec_layer(["Be good!", "Question: be or be?"]))

# using pre-trained language model components
hub_layer = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim50/2")
sentence_embeddings = hub_layer(tf.constant(["To be", "Not to be"]))
print(sentence_embeddings.numpy().round(2))

# image pre-processing layers
images = load_sample_images()["images"]
crop_image_layer = tfk.layers.CenterCrop(height=100, width=100)
cropped_images = crop_image_layer(images)

#%% the tensorflow datasets (TFDS) project
import tensorflow_datasets as tfds

datasets = tfds.load(name="mnist")
mnist_train, mnist_test = datasets["train"], datasets["test"]

for batch in mnist_train.shuffle(10000, seed=42).batch(32).prefetch(1):
    images = batch["image"]
    labels = batch["label"]

mnist_train = mnist_train.shuffle(10000, seed=42).batch(32)
mnist_train = mnist_train.map(lambda items: (items["image"], items["label"]))    
mnist_train = mnist_train.prefetch(1)

train_set, val_set, test_set = tfds.load(
    name="mnist", split=["train[:90%]", "train[90%:]", "test"], as_supervised=True)
train_set = train_set.shuffle(buffer_size=10000, seed=42).batch(32).prefetch(1)
val_set = val_set.batch(32).cache()
test_set = test_set.batch(32).cache()

tfk.backend.clear_session()
tf.random.set_seed(42)
model = tfk.Sequential([
    tfk.layers.Flatten(input_shape=(28, 28)),
    tfk.layers.Dense(10, activation="softmax")
    ])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
history = model.fit(train_set, validation_data=val_set, epochs=5)
test_loss, test_accuracy = model.evaluate(test_set)
print(test_loss, test_accuracy)

#%% Coding Exercises: Exercise 9
import tensorflow.keras as tfk
import tensorflow as tf
from tensorflow.train import Example, Features, Feature, BytesList, Int64List
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# load the fashion mnist dataset, split it into a training, validation and test set
(X_train_val, y_train_val), (X_test, y_test) = tfk.datasets.fashion_mnist.load_data()
X_val, X_train = X_train_val[:5000], X_train_val[5000:]
y_val, y_train = y_train_val[:5000], y_train_val[5000:]

# shuffle the training set and save each dataset to multiple TFRecord files
# each record should be a serialised Example protobuff with two features: the serialised
# image (use tf.io.serialize_tensor() to serialise each image, and the label
tf.random.set_seed(42)
train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_set = train_set.shuffle(len(X_train), seed=42)
val_set = tf.data.Dataset.from_tensor_slices((X_val, y_val))
test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test))

def create_example(image, label):
    image_data = tf.io.serialize_tensor(image)
    example = Example(
        features=Features(
            feature={
                "image": Feature(bytes_list=BytesList(value=[image_data.numpy()])),
                "label": Feature(int64_list=Int64List(value=[label])),
            }))
    return example

def write_tfrecords(name, dataset, num_shards=10):
    paths = ["./datasets/ch13_ex09/{}.tfrecord-{:03d}".format(name, index)
             for index in range(num_shards)]
    writers = [tf.io.TFRecordWriter(path) for path in paths]
    for index, (image, label) in dataset.enumerate():
        shard = index % num_shards
        example = create_example(image, label)
        writers[shard].write(example.SerializeToString())

    return paths

train_paths = write_tfrecords("train", train_set)
val_paths = write_tfrecords("val", val_set)
test_paths = write_tfrecords("test", test_set)

# use tf.data to create an efficient dataset for each set
def preprocess(tfrecord):
    feature_description = {
        "image": tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=""),
        "label": tf.io.FixedLenFeature(shape=[], dtype=tf.int64, default_value=-1)
        }
    example = tf.io.parse_single_example(tfrecord, feature_description)
    image = tf.io.parse_tensor(example["image"], out_type=tf.uint8)
    image = tf.reshape(image, shape=[28, 28])
    return image, example["label"]

def fashion_mnist_dataset(filepaths, batch_size=32, cache=True):
    dataset = tf.data.TFRecordDataset(filepaths)
    if cache:
        dataset = dataset.cache()
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)

train_set2 = fashion_mnist_dataset(train_paths)
val_set2 = fashion_mnist_dataset(val_paths)
test_set2 = fashion_mnist_dataset(test_paths)

for X, y in train_set2.take(1):
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(np.array(X[i]), cmap="binary")
        plt.axis("off")
        plt.title(np.arry(str(y[i])))


# use a keras model to train the datasets, including a preprocessing layer to standardise
# each input feature
tf.random.set_seed(42)
norm_layer = tfk.layers.Normalization(input_shape=(28, 28))
image_batches = train_set.take(100).map(lambda image, label: image)
images = np.concatenate(list(image_batches.as_numpy_iterator()), axis=0).astype(
    np.float32)
norm_layer.adapt(images)

tfk.backend.clear_session()
model = tfk.Sequential()
model.add(norm_layer)
model.add(tfk.layers.Flatten())
model.add(tfk.layers.Dense(100, activation="relu"))
model.add(tfk.layers.Dense(10, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])

log_dir = "./logs/ch13_ex09/run_" + datetime.now().strftime("%Y-%m-%d")
tensorboard_cb = tfk.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,
                                           profile_batch=10)
history = model.fit(train_set2, validation_data=val_set2, epochs=10,
                    callbacks=[tensorboard_cb])
model.evaluate(val_set2)

#%% Coding Exercises: Exercise 10
import tensorflow.keras as tfk
import os
import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# download the Large Movie Review dataset. The data is contained in two directories:
# train and test, each containing a pos sub-directory with 12,500 positive reviews and a
# neg sub-directory containing 12,500 negative reviews. Each review is stored in a
# separate text file
filename = "aclImdb_v1.tar.gz"
filepath = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tfk.utils.get_file(filename, filepath, extract=True, cache_dir="~/./datasets")

source_dir = "E:/tmp/.keras/datasets/"
target_dir = "./datasets"
os.makedirs(target_dir, exist_ok=True)
for file in os.listdir("E:/tmp/.keras/datasets/"):
    source_path = os.path.join(source_dir, file)
    target_path = os.path.join(target_dir, file)
    shutil.move(source_path, target_path)
    
def get_review_paths(train_test, pos_neg, root_dir="./datasets/aclImdb"):
    return [str(path) for path in Path(root_dir, train_test, pos_neg).glob("*.txt")]

train_pos = get_review_paths("train", "pos")
train_neg = get_review_paths("train", "neg")
test_val_pos = get_review_paths("test", "pos")
test_val_neg = get_review_paths("test", "neg")

print(len(train_pos), len(train_neg))
print(len(test_val_pos), len(test_val_neg))

# split the test set into a validation set (15,000 reviews) and a test set (10,000
# reviews)
np.random.shuffle(test_val_pos)
val_pos = test_val_pos[:7500]
test_pos = test_val_pos[7500:]

np.random.shuffle(test_val_neg)
val_neg = test_val_neg[:7500]
test_neg = test_val_neg[7500:]

# use tf.data to create an efficient datset for each set
def create_dataset(neg_filepaths, pos_filepaths):
    reviews = []
    labels = []
    for filepaths, label in ((neg_filepaths, 0), (pos_filepaths, 1)):
        for filepath in filepaths:
            with open(filepath, encoding="utf-8") as review_file:
                reviews.append(review_file.read())
            labels.append(label)
            
    combined_data = list(zip(reviews, labels))
    np.random.shuffle(combined_data)
            
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.constant([review for review, _ in combined_data]),
         tf.constant([label for _, label in combined_data])))
    return dataset

train_set = create_dataset(train_neg, train_pos).batch(32).prefetch(1)
val_set = create_dataset(val_neg, val_pos).batch(32).prefetch(1)
test_set = create_dataset(test_neg, test_pos).batch(32).prefetch(1)

for X, y in test_set:
    print(y)
    print()

# create a binary classification model using a TextVectorization layer to preprocess each
# review
MAX_FEATURES = 1000
text_vec_layer = tfk.layers.TextVectorization(max_tokens=MAX_FEATURES,
                                              output_mode="tf_idf")

def map_dataset(dataset):
    reviews = dataset.map(lambda review, label: review)
    return reviews

train_reviews = map_dataset(train_set)
val_reviews = map_dataset(val_set)
test_reviews = map_dataset(test_set)

text_vec_layer.adapt(train_reviews)

tf.random.set_seed(42)
tfk.backend.clear_session()
model = tfk.models.Sequential()
model.add(text_vec_layer)
model.add(tfk.layers.Dense(100, activation="relu"))
model.add(tfk.layers.Dense(1, activation="sigmoid"))
model.summary()

model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])
model.fit(train_set, epochs=10, validation_data=val_set)

# use TFDS to load the same dataset more easily: tfds.load("imdb_reviews")
datasets = tfds.load(name="imdb_reviews")
imdb_train, imdb_test = datasets["train"], datasets["test"]