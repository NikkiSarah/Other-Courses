#%% forecasting a time series
import tensorflow.keras as tfk
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
import numpy as np

tfk.utils.get_file(
    "ridership.tgz",
    "https://github.com/ageron/data/raw/main/ridership.tgz",
    cache_dir=".",
    extract=True
)

path = Path("datasets/ridership/CTA_-_Ridership_-_Daily_Boarding_Totals.csv")
df = pd.read_csv(path, parse_dates=["service_date"])
df.columns = ["date", "day_type", "bus", "rail", "total"]
df = df.sort_values("date").set_index("date")
df = df.drop("total", axis=1)
df = df.drop_duplicates()
df.head()

df["2019-03":"2019-05"].plot(marker=".", )
plt.legend(loc="lower left")

diff_7 = df[["bus", "rail"]].diff(7)["2019-03":"2019-05"]

fig, axs = plt.subplots(2, 1, sharex=True)
df.plot(ax=axs[0], legend=False, marker=".")
df.shift(7).plot(ax=axs[0], legend=False, linestyle=":")
diff_7.plot(ax=axs[1], marker=".")

print(list(df.loc["2019-05-25":"2019-05-27"]["day_type"]))
print(diff_7.abs().mean())

targets = df[["bus", "rail"]]["2019-03":"2019-05"]
print((diff_7 / targets).abs().mean())

period = slice("2001", "2019")
df_monthly = df.drop('day_type', axis=1).resample("M").mean()
rolling_avg_12_months = df_monthly[period].rolling(window=12).mean()

fig, ax = plt.subplots()
df_monthly[period].plot(ax=ax, marker=".")
rolling_avg_12_months.plot(ax=ax, legend=False)

df_monthly.diff(12)[period].plot(marker=".")

# the ARMA model family
origin, today = "2019-01-01", "2019-05-31"
rail_series = df.loc[origin:today]["rail"].asfreq("D")
model = ARIMA(rail_series, order=(1, 0, 0), seasonal_order=(0, 1, 1, 7))
model = model.fit()
y_pred = model.forecast()
print(y_pred)

origin, start_date, end_date = "2019-01-01", "2019-03-01", "2019-05-31"
time_period = pd.date_range(start_date, end_date)
rail_series = df.loc[origin:end_date]["rail"].asfreq("D")
y_preds = []
for today in time_period.shift(-1):
    model = ARIMA(rail_series[origin:today], order=(1, 0, 0), seasonal_order=(0, 1, 1, 7))
    model = model.fit()
    y_pred = model.forecast()[0]
    y_preds.append(y_pred)
    
y_preds = pd.Series(y_preds, index=time_period)
mae = (y_preds - rail_series[time_period]).abs().mean()
print(mae)

# preparing the data for ML models
my_series = [0, 1, 2, 3, 4, 5]
my_dataset = tfk.utils.timeseries_dataset_from_array(
    my_series,
    targets=my_series[3:],
    sequence_length=3,
    batch_size=2
    )
print(list(my_dataset))

for window_dataset in tf.data.Dataset.range(6).window(4, shift=1):
    for ele in window_dataset:
        print(f"{ele}", end=" ")
    print()

dataset = tf.data.Dataset.range(6).window(4, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window_dataset: window_dataset.batch(4))
for window_tensor in dataset:
    print(f"{window_tensor}")
    
def to_windows(dataset, length):
    dataset = dataset.window(length, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window_ds: window_ds.batch(length))
    return dataset

dataset = to_windows(tf.data.Dataset.range(6), 4)
dataset = dataset.map(lambda window: (window[:-1], window[-1]))
print(list(dataset.batch(2)))

rail_train = df["rail"]["2016-01":"2018-12"] / 1e6
rail_val = df["rail"]["2019-01":"2019-05"] / 1e6
rail_test = df["rail"]["2019-06":] / 1e6

seq_length = 56
train_ds = tfk.utils.timeseries_dataset_from_array(
    rail_train.to_numpy(),
    targets=rail_train[seq_length:],
    sequence_length=seq_length,
    batch_size=32,
    shuffle=True,
    seed=42)
val_ds = tfk.utils.timeseries_dataset_from_array(
    rail_val.to_numpy(),
    targets=rail_val[seq_length:],
    sequence_length=seq_length,
    batch_size=32)

# forecasting using a linear model
tf.random.set_seed(42)
model = tfk.Sequential([
    tfk.layers.Dense(1, input_shape=[seq_length])
    ])

early_stopping_cb = tfk.callbacks.EarlyStopping(monitor="val_mae", patience=50,
                                                restore_best_weights=True)
optimiser = tfk.optimizers.SGD(learning_rate=0.02, momentum=0.9)
model.compile(loss=tfk.losses.Huber(), optimizer=optimiser, metrics=["mae"])
history = model.fit(train_ds, validation_data=val_ds, epochs=500,
                    callbacks=[early_stopping_cb])
val_loss, val_mae = model.evaluate(val_ds)
print(val_mae * 1e6)

# forecasting using a simple RNN
tfk.backend.clear_session()
tf.random.set_seed(42)
model = tfk.Sequential([
    tfk.layers.SimpleRNN(1, input_shape=[None, 1])
    ])

def fit_and_evaluate_model(model, train_set, val_set, learning_rate, epochs=500):
    early_stopping_cb = tfk.callbacks.EarlyStopping(monitor="val_mae", patience=50,
                                                    restore_best_weights=True)
    optimiser = tfk.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(loss=tfk.losses.Huber(), optimizer=optimiser, metrics=["mae"])
    history = model.fit(train_set, validation_data=val_set, epochs=epochs,
                        callbacks=[early_stopping_cb])
    val_loss, val_mae = model.evaluate(val_set)
    print(val_mae * 1e6)
    
fit_and_evaluate_model(model, train_ds, val_ds, learning_rate=0.02)


tfk.backend.clear_session()
tf.random.set_seed(42)
univar_model = tfk.Sequential([
    tfk.layers.SimpleRNN(32, input_shape=[None, 1]),
    tfk.layers.Dense(1)
    ])

fit_and_evaluate_model(univar_model, train_ds, val_ds, learning_rate=0.05)

# forecasting using a deep RNN
tfk.backend.clear_session()
tf.random.set_seed(42)
deep_model = tfk.Sequential([
    tfk.layers.SimpleRNN(32, return_sequences=True, input_shape=[None, 1]),
    tfk.layers.SimpleRNN(32, return_sequences=True),
    tfk.layers.SimpleRNN(32),
    tfk.layers.Dense(1)
    ])

fit_and_evaluate_model(deep_model, train_ds, val_ds, learning_rate=0.01)

# forecasting multivariate time series
df_mulvar = df[["bus", "rail"]] / 1e6
df_mulvar["next_day_type"] = df.day_type.shift(-1)
df_mulvar = pd.get_dummies(df_mulvar, dtype=float)

mulvar_train = df_mulvar["2016-01":"2018-12"]
mulvar_val = df_mulvar["2019-01":"2019-05"]
mulvar_test = df_mulvar["2019-06":]

train_mulvar_ds = tfk.utils.timeseries_dataset_from_array(
    mulvar_train.to_numpy(),
    targets=mulvar_train["rail"][seq_length:],
    sequence_length=seq_length,
    batch_size=32,
    shuffle=True,
    seed=42
)
val_mulvar_ds = tfk.utils.timeseries_dataset_from_array(
    mulvar_val.to_numpy(),
    targets=mulvar_val["rail"][seq_length:],
    sequence_length=seq_length,
    batch_size=32
)

tfk.backend.clear_session()
mulvar_model = tfk.Sequential([
    tfk.layers.SimpleRNN(32, input_shape=[None, 5]),
    tfk.layers.Dense(1)
])

fit_and_evaluate_model(mulvar_model, train_mulvar_ds, val_mulvar_ds, learning_rate=0.05)
rail_naive = mulvar_val["rail"].shift(7)[seq_length:]
rail_target = mulvar_val["rail"][seq_length:]
print((rail_target - rail_naive).abs().mean() * 1e6)


train_multask_ds = tfk.utils.timeseries_dataset_from_array(
    mulvar_train.to_numpy(),
    targets=mulvar_train[["bus", "rail"]][seq_length:],
    sequence_length=seq_length,
    batch_size=32,
    shuffle=True,
    seed=42)
val_multask_ds = tfk.utils.timeseries_dataset_from_array(
    mulvar_val.to_numpy(),
    targets=mulvar_val[["bus", "rail"]][seq_length:],
    sequence_length=seq_length,
    batch_size=32)

tfk.backend.clear_session()
tf.random.set_seed(42)
multask_model = tfk.Sequential([
    tfk.layers.SimpleRNN(32, input_shape=[None, 5]),
    tfk.layers.Dense(2)])

fit_and_evaluate_model(multask_model, train_multask_ds, val_multask_ds,
                       learning_rate=0.02)
y_preds_val = multask_model.predict(val_multask_ds)
for idx, name in enumerate(["bus", "rail"]):
    mae = 1e6 * tfk.metrics.mean_absolute_error(mulvar_val[name][seq_length:],
                                                y_preds_val[:, idx])
    print(name, int(mae))


# forecasting several time steps ahead
X = rail_val.to_numpy()[np.newaxis, :seq_length, np.newaxis]
for step_ahead in range(14):
    y_pred_one = univar_model.predict(X)
    X = np.concatenate([X, y_pred_one.reshape(1, 1, 1)], axis=1)

def split_inputs_and_targets(mulvar_series, ahead=14, target_col=1):
    return mulvar_series[:, :-ahead], mulvar_series[:, -ahead:, target_col]

ahead_train_ds = tfk.utils.timeseries_dataset_from_array(
    mulvar_val.to_numpy(),
    targets=None,
    sequence_length=seq_length + 14,
    batch_size=32).map(split_inputs_and_targets)
ahead_val_ds = tfk.utils.timeseries_dataset_from_array(
    mulvar_train.to_numpy(),
    targets=None,
    sequence_length=seq_length + 14,
    batch_size=32).map(split_inputs_and_targets)

tfk.backend.clear_session()
tf.random.set_seed(42)
ahead_model = tfk.Sequential([
    tfk.layers.SimpleRNN(32, input_shape=[None, 5]),
    tfk.layers.Dense(14)
    ])

fit_and_evaluate_model(ahead_model, ahead_train_ds, ahead_val_ds, learning_rate=0.02)

X = mulvar_val.to_numpy()[np.newaxis, :seq_length]
y_pred = ahead_model.predict(X)
print(y_pred)

# forecasting using a sequence-to-sequence model
my_series = tf.data.Dataset.range(7)
dataset = to_windows(to_windows(my_series, 3), 4)
print(list(dataset))

dataset = dataset.map(lambda S: (S[:, 0], S[:, 1:]))
print(list(dataset))

def to_seq2seq_dataset(series, seq_length=56, ahead=14, target_col=1, batch_size=32,
                       shuffle=False, seed=None):
    ds = to_windows(tf.data.Dataset.from_tensor_slices(series), ahead + 1)
    ds = to_windows(ds, seq_length).map(lambda S: (S[:, 0], S[:, 1:, 1]))
    if shuffle:
        ds = ds.shuffle(8 * batch_size, seed=seed)
    return ds.batch(batch_size)

seq2seq_train = to_seq2seq_dataset(mulvar_train, shuffle=True, seed=42)
seq2seq_val = to_seq2seq_dataset(mulvar_val)

tfk.backend.clear_session()
tf.random.set_seed(42)
seq2seq_model = tfk.Sequential([
    tfk.layers.SimpleRNN(32, return_sequences=True, input_shape=[None, 5]),
    tfk.layers.Dense(14)
    ])
fit_and_evaluate_model(seq2seq_model, seq2seq_train, seq2seq_val, learning_rate=0.1)

X = mulvar_val.to_numpy()[np.newaxis, :seq_length]
y_pred_14 = seq2seq_model.predict(X)[0, -1]

y_pred_val = seq2seq_model.predict(seq2seq_val)
for ahead in range(14):
    preds = pd.Series(y_pred_val[:-1, -1, ahead],
                      index=mulvar_val.index[56 + ahead : -14 + ahead])
    mae = (preds - mulvar_val["rail"]).abs().mean() * 1e6
    print(f"MAE for +{ahead + 1}: {mae:,.0f}")

#%% handling long sequences

# fighting the unstable gradients problem
class LNSimpleRNNCell(tfk.layers.Layer):
    def __init__(self, units, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.state_size = units
        self.output_size = units
        self.simple_rnn_cell = tfk.layers.SimpleRNNCell(units, activation=None)
        self.layer_norm = tfk.layers.LayerNormalization()
        self.activation = tfk.activations.get(activation)
        
    def call(self, inputs, states):
        outputs, new_states = self.simple_rnn_cell(inputs, states)
        norm_outputs = self.activation(self.layer_norm(outputs))
        return norm_outputs, [norm_outputs]

tfk.backend.clear_session()
tf.random.set_seed(42)
custom_ln_model= tfk.Sequential([
    tfk.layers.RNN(LNSimpleRNNCell(32), return_sequences=True, input_shape=[None, 5]),
    tfk.layers.Dense(14)
    ])

fit_and_evaluate_model(custom_ln_model, seq2seq_train, seq2seq_val, learning_rate=0.1,
                       epochs=5)

# tackling the short-term memory problem
tfk.backend.clear_session()
tf.random.set_seed(42)
lstm_model = tfk.Sequential([
    tfk.layers.LSTM(32, return_sequences=True, input_shape=[None, 5]),
    tfk.layers.Dense(14)
    ])

fit_and_evaluate_model(lstm_model, seq2seq_train, seq2seq_val, learning_rate=0.1,
                       epochs=5)


tfk.backend.clear_session()
tf.random.set_seed(42)
conv_rnn_model = tfk.Sequential([
    tfk.layers.Conv1D(filters=32, kernel_size=4, strides=2, activation="relu",
                      input_shape=[None, 5]),
    tfk.layers.GRU(32, return_sequences=True),
    tfk.layers.Dense(14)
    ])

longer_train = to_seq2seq_dataset(mulvar_train, seq_length=112, shuffle=True, seed=42)
longer_val = to_seq2seq_dataset(mulvar_val, seq_length=112)
downsampled_train = longer_train.map(lambda X, Y: (X, Y[:, 3::2]))
downsampled_val = longer_val.map(lambda X, Y: (X, Y[:, 3::2]))

fit_and_evaluate_model(conv_rnn_model, downsampled_train, downsampled_val,
                       learning_rate=0.1, epochs=5)


#%% Coding Exercises: Exercise 9
import tensorflow.keras as tfk
import os
from pathlib import Path
import tensorflow as tf
from tensorflow.data import TFRecordDataset
import matplotlib.pyplot as plt
import numpy as np

# train a classification model for the SketchRNN dataset
tf_download_root = "http://download.tensorflow.org/data/"
filename = "quickdraw_tutorial_dataset_v1.tar.gz"
filepath = tfk.utils.get_file(filename,
                              tf_download_root + filename,
                              cache_dir=".",
                              extract=True)

root_dir = "./datasets/quickdraw"
print(root_dir)

with open(os.path.join(root_dir, "eval.tfrecord.classes")) as class_file:
    val_classes = class_file.readlines()

with open(os.path.join(root_dir, "training.tfrecord.classes")) as class_file:
    train_classes = class_file.readlines()

assert train_classes == val_classes

classes = [name.strip().lower() for name in train_classes]
print(len(classes))

train_paths = [str(path) for path in Path(root_dir).glob("training.tfrecord-*")][:5]
val_paths = [str(path) for path in Path(root_dir).glob("eval.tfrecord-*")][:5]

def parse_examples(data_batch):
    feature_descriptions = {
        "ink": tf.io.VarLenFeature(dtype=tf.float32),
        "shape": tf.io.FixedLenFeature(shape=[2], dtype=tf.int64),
        "class_index": tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
    }
    examples = tf.io.parse_example(data_batch, feature_descriptions)
    
    flat_sketches = tf.sparse.to_dense(examples["ink"])
    sketches = tf.reshape(flat_sketches, shape=[tf.shape(data_batch)[0], -1, 3])
    lengths = examples["shape"][:, 0]
    labels = examples["class_index"][:, 0]
    
    return sketches, lengths, labels


def quickdraw_dataset(filepaths, batch_size=32, shuffle_buffer_size=None,
                      n_parse_threads=5, n_read_threads=5, cache=False):
    dataset = TFRecordDataset(filepaths, num_parallel_reads=n_read_threads)
    if cache:
        dataset = dataset.cache()
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_examples, num_parallel_calls=n_parse_threads)
    return dataset.prefetch(1)

train_set = quickdraw_dataset(train_paths, shuffle_buffer_size=10000)
val_set = quickdraw_dataset(val_paths)

for sketches, lengths, labels in train_set.take(1):
    print(sketches)
    print('-----')
    print(lengths)
    print('-----')
    print(labels)

def draw_sketch(sketch, label=None, color='gray'):
    origin = np.array([[0., 0., 0.]])
    sketch = np.r_[origin, sketch]
    stroke_end_indices = np.argwhere(sketch[:, -1] == 1.)[:, 0]
    coordinates = sketch[:, :2].cumsum(axis=0)
    strokes = np.split(coordinates, stroke_end_indices + 1)
    title = classes[label.numpy()]
    plt.title(title)
    for stroke in strokes:
        plt.plot(stroke[:, 0], -stroke[:, 1], ".-", color=color)
    plt.axis("off")

def draw_sketches(sketches, lengths, labels):
    n_cols = 3
    n_rows = n_cols
    for idx in range(n_cols * n_rows):
        sketch, length, label = sketches[idx], lengths[idx], labels[idx]
        plt.subplot(n_rows, n_cols, idx+1)
        draw_sketch(sketch[:length], label=label)
    plt.show()

for sketches, lengths, labels in train_set.take(1):
    draw_sketches(sketches, lengths, labels)

# check the distribution of lengths
num_train_batches = len(list(train_set))
num_val_batches = len(list(val_set))

sketch_lengths = np.concatenate([lengths for _, lengths, _ in
                                 train_set.take(num_train_batches)])
plt.hist(sketch_lengths, bins=150, density=True)
plt.axis([0, 200, 0, 0.021])
plt.xlabel("length")
plt.ylabel("density")

def crop_long_sketches(dataset, max_length=100):
    cropped_dataset = dataset.map(lambda sketches, lengths, labels:
                                  (sketches[:, :max_length], labels))
    return cropped_dataset

cropped_train_set = crop_long_sketches(train_set)
cropped_val_set = crop_long_sketches(val_set)

num_conv_layers = 3
conv_filters = [48, 64, 96]
conv_kernels = [5, 5, 3]
num_rnn_layers = 3
batch_normalisation = False
lr = 0.0001
gradient_cv = 9.0
optimiser = [tfk.optimizers.Adam(learning_rate=lr, clipvalue=gradient_cv),
             tfk.optimizers.AdamW(learning_rate=lr, clipvalue=gradient_cv),
             tfk.optimizers.experimental.Nadam(learning_rate=lr, clipvalue=gradient_cv)]
num_epochs = 5

tfk.backend.clear_session()
tf.random.set_seed(42)
model = tfk.Sequential()
for i in range(num_conv_layers):
    model.add(tfk.layers.Conv1D(filters=conv_filters[i],kernel_size=conv_kernels[i],
                                activation="swish", strides=1, padding="same"))
    if batch_normalisation is True:
        model.add(tfk.layers.BatchNormalization())
    else:
        model.add(tfk.layers.Dropout(0.3))
for i in range(num_rnn_layers-1):
    model.add(tfk.layers.LSTM(128, return_sequences=True, return_state=True))
model.add(tfk.layers.LSTM(128))
model.add(tfk.layers.Dense(len(classes), activation="softmax"))

model.compile(optimizer=optimiser[0], loss="sparse_categorical_crossentropy",
              metrics=["accuracy", "sparse_top_k_categorical_accuracy"])

history = model.fit(cropped_train_set, epochs=num_epochs, verbose=2, 
                    validation_data=cropped_val_set)

model.summary()

def plot_fit_history(fit_history):
    history_df = pd.DataFrame(fit_history.history)
    fig, axs = plt.subplots(2, 1, sharex=True)
    history_df[["loss", "val_loss"]].plot(ax=axs[0], style=["b-", "g--."])
    history_df[["accuracy", "val_accuracy"]].plot(ax=axs[1], style=["b-", "g--."])
    
plot_fit_history(history)

y_test = np.concatenate([labels for _, _, labels in val_set])
y_probas = model.predict(val_set)

print(np.mean(tfk.metrics.sparse_top_k_categorical_accuracy(y_test, y_probas)))

n_new = 10
y_probas = model.predict(sketches)
top_k = tf.nn.top_k(y_probas, k=5)
for index in range(n_new):
    draw_sketch(sketches[index])
    print("Top-5 predictions:".format(index + 1))
    for k in range(5):
        class_ = classes[top_k.indices[index, k]]
        proba = 100 * top_k.values[index, k]
        print("  {}. {} {:.3f}%".format(k + 1, class_, proba))
    print("Answer: {}".format(classes[labels[index].numpy()]))


#%% Coding Exercises: Exercise 10
import tensorflow.keras as tfk
import tarfile
import os
from pathlib import Path
import pandas as pd
import numpy as np
import random
from music21 import converter, meter, midi, note, stream, tempo
from pygame import mixer
import tensorflow as tf

# download the Bach chorales datset and unzip it
tf_download_root = "https://github.com/ageron/data/raw/main/"
filename = "jsb_chorales.tgz"
# filepath = tfk.utils.get_file(filename,
#                               tf_download_root + filename,
#                               cache_dir=".",
#                               extract=True)

def extract_tgz(tgz_file, target_dir):
    with tarfile.open(tgz_file, 'r:gz') as f:
        f.extractall(target_dir)

# extract_tgz(os.path.join('./datasets', filename), './datasets')

root_dir = './datasets/jsb_chorales'
print(root_dir)

paths_train = sorted([str(path) for path in Path(root_dir).glob("train/chorale_*")])
paths_val = sorted([str(path) for path in Path(root_dir).glob("valid/chorale_*")])
paths_test = sorted([str(path) for path in Path(root_dir).glob("test/chorale_*")])

def load_chorales(filepaths):
    return [pd.read_csv(path_).values.tolist() for path_ in filepaths]

chorales_train_raw = load_chorales(paths_train)
chorales_val_raw = load_chorales(paths_val)
chorales_test_raw = load_chorales(paths_test)


def select_chorale(chorale_set, shorter_chorales=False, num_chords=None, save_path=None):
    if shorter_chorales:
        chorale_lengths = [len(chorale) for chorale in chorale_set]
        shorter_chorales = [chorale for chorale in chorale_set
                            if len(chorale) < np.median(chorale_lengths)]
   
        chorale_idx = random.randint(0, len(shorter_chorales) - 1)
        chorale = shorter_chorales[chorale_idx]
    else:
        chorale_idx = random.randint(0, len(chorale_set) - 1)
        chorale = chorale_set[chorale_idx]
    
    if num_chords is not None:
        chorale = chorale[:num_chords]
    
    chorale_stream = stream.Score()
    chorale_part = stream.Part()
    chorale_stream.append(chorale_part)
    chorale_notes = [note.Note(chorale_note) for chord in chorale
                    for chorale_note in chord]
    chorale_part.append(chorale_notes)

    chorale_stream.append(meter.TimeSignature('4/4'))
    chorale_stream.append(tempo.MetronomeMark(number=160))
    
    if save_path is not None:
        midi_save_path=save_path
        chorale_stream.write(fmt='midi', fp=midi_save_path)
    
    return chorale, chorale_idx


def play_chorale(midi_file_path):
    mixer.init()
    mixer.music.load(midi_file_path)
    mixer.music.play()
    
    while mixer.music.get_busy():
        continue
    # mixer.quit()

midi_save_path='./outputs/chorale.mid'
# random_chorale, random_chorale_idx = select_chorale(chorales_train_raw, num_chords=10,
#                                                     save_path=midi_save_path)
play_chorale(midi_save_path)

# check minimum and maximum values
chorales_all = chorales_train_raw + chorales_val_raw + chorales_test_raw
chorales_all_flat = [chorale_note for chorale in chorales_all for chord in chorale for 
                     chorale_note in chord]
unique_notes = set(chorales_all_flat)

min_note = min(unique_notes - {0})    
max_note = max(unique_notes)
print(min_note, max_note)

# pre-process the dataset such that the target is only a single note rather than an
# entire chord
def process_chorales(in_chorales, window_size=32, window_shift=15, cache=True,
                     shuffle_buffer_size=None, batch_size=32):
    ragged_chorale = tf.ragged.constant(in_chorales, ragged_rank=1)
    tensor_slice_ds = tf.data.Dataset.from_tensor_slices(ragged_chorale)
    
    flat_ds = tensor_slice_ds.flat_map(
        lambda chorale: tf.data.Dataset.from_tensor_slices(chorale))
    window_ds = flat_ds.window(size=window_size+1, shift=window_shift,
                               drop_remainder=True)
    batch_ds = window_ds.flat_map(lambda window: window.batch(window_size+1).map(
        lambda w: tf.reshape(w, [-1])))
    shift_ds = batch_ds.map(lambda note: tf.where(note == 0, note, note - min_note+1))

    processed_chorales = shift_ds
    
    if cache:
        processed_chorales = processed_chorales.cache()
    if shuffle_buffer_size:
        processed_chorales = processed_chorales.shuffle(shuffle_buffer_size)
    
    batched_chorales = processed_chorales.batch(batch_size)
    batched_chorales = batched_chorales.map(lambda batch: (batch[:, :-1], batch[:, 1:]))
    
    return batched_chorales


chorales_train = process_chorales(chorales_train_raw + chorales_test_raw,
                                  shuffle_buffer_size=1000)
chorales_val = process_chorales(chorales_val_raw)

for item in chorales_train.take(1):
    X = item[0].numpy()
    y = item[1].numpy()
    
    print("X:", item[0])
    print("y:", item[1])
    print("shape:", item[0].shape)
    
# train a model - recurrent, convlutional or both - that can predict the next time step
# (4 notes), given a sequence of time steps from a chorale
# https://www.kaggle.com/code/s4vyss/recurrentneuralnetworks-chapter-15
num_notes = len(unique_notes)
num_embedding_dims = int([item[0].shape[1] for item in chorales_train.take(1)][0])
conv_filters = [32, 48, 64, 96]
dilations = [1, 2, 4, 8]
lr = 1e-3

tfk.backend.clear_session()
tf.random.set_seed(42)
model = tfk.Sequential()
model.add(tfk.layers.Embedding(input_dim=num_notes, output_dim=num_embedding_dims,
                               input_shape=[None]))
for i in range(4):
    model.add(tfk.layers.Conv1D(conv_filters[i], kernel_size=2, padding="causal",
                                activation="relu", dilation_rate=dilations[i]))
    model.add(tfk.layers.BatchNormalization())
model.add(tfk.layers.LSTM(256, return_sequences=True))
model.add(tfk.layers.Dense(num_notes, activation="softmax"))

model.summary()

model.compile(loss="sparse_categorical_crossentropy", 
              optimizer=tfk.optimizers.Nadam(learning_rate=lr), metrics=["accuracy"])
early_stopping_cb = tfk.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                                restore_best_weights=True)
history = model.fit(chorales_train, validation_data=chorales_val, epochs=50,
                    callbacks=[early_stopping_cb])

def plot_fit_history(fit_history):
    history_df = pd.DataFrame(fit_history.history)
    fig, axs = plt.subplots(2, 1, sharex=True)
    history_df[["loss", "val_loss"]].plot(ax=axs[0], style=["b-", "g--."])
    history_df[["accuracy", "val_accuracy"]].plot(ax=axs[1], style=["b-", "g--."])

plot_fit_history(history)

model.evaluate(chorales_val)
model.save("./outputs/ch15_ex10_conv_model")


tfk.backend.clear_session()
tf.random.set_seed(42)
model = tfk.models.Sequential()
model.add(tfk.layers.Embedding(input_dim=num_notes, output_dim=num_embedding_dims,
                               input_shape=[None]))
for _ in range(3):
    model.add(tfk.layers.GRU(512, return_sequences=True))
    model.add(tfk.layers.GroupNormalization())
model.add(tfk.layers.Dense(num_notes, activation="softmax"))

model.summary()

model.compile(loss="sparse_categorical_crossentropy", 
              optimizer=tfk.optimizers.Nadam(learning_rate=lr), metrics=["accuracy"])
history = model.fit(chorales_train, validation_data=chorales_val, epochs=50,
                    callbacks=[early_stopping_cb])
   
plot_fit_history(history)

model.evaluate(chorales_val)
model.save("./outputs/ch15_ex10_gru_rnn_model")


tfk.backend.clear_session()
tf.random.set_seed(42)
model = tfk.models.Sequential()
model.add(tfk.layers.Embedding(input_dim=num_notes, output_dim=num_embedding_dims,
                               input_shape=[None]))
for _ in range(3):
    model.add(tfk.layers.LSTM(512, return_sequences=True))
    model.add(tfk.layers.GroupNormalization())
model.add(tfk.layers.Dense(num_notes, activation="softmax"))

model.summary()

model.compile(loss="sparse_categorical_crossentropy", 
              optimizer=tfk.optimizers.Nadam(learning_rate=lr), metrics=["accuracy"])
history = model.fit(chorales_train, validation_data=chorales_val, epochs=50,
                    callbacks=[early_stopping_cb])
   
plot_fit_history(history)

model.evaluate(chorales_val)
model.save("./outputs/ch15_ex10_lstm_rnn_model")

# use the model to generate Bach-like music, one note at a time:
chorale_model = tfk.models.load_model("./outputs/ch15_ex10_gru_rnn_model")
chorale_length = 10

midi_save_path = './outputs/seed_chords.mid'
# seed_chords, _ = select_chorale(chorales_test_raw, num_chords=8, save_path=midi_save_path)
# play_chorale(midi_save_path)

def convert_midi_file(midi_file_path):
    midi_stream = converter.parse(midi_file_path)
    
    midi_notes = []
    for ele in midi_stream.flatten():
        if isinstance(ele, note.Note):
            midi_notes.append(ele.pitch.midi)
    
    chords = [midi_notes[i:i+4] for i in range(0, len(midi_notes), 4)]
    
    return chords

seed_chords = convert_midi_file(midi_save_path)
play_chorale(midi_save_path)

seed_tensor = tf.constant(seed_chords, dtype=tf.int64)
shifted_tensor = tf.where(seed_tensor == 0, seed_tensor, seed_tensor - min_note + 1)
# seed_arpegio = tf.reshape(shifted_tensor, [1, -1])

# for chord in range(chorale_length):
#     for note in range(4):
#         next_notes = chorale_model.predict(seed_arpegio, verbose=0).argmax(axis=-1)
#         next_note = next_notes[:, -1:]
#         seed_arpegio = tf.concat([seed_arpegio, next_note], axis=1)
# shifted_arpegio = tf.where(seed_arpegio == 0, seed_arpegio, seed_arpegio + min_note - 1)
# reshaped_arpegio = tf.reshape(shifted_arpegio, shape=[-1, 4])
# generated_chorale = reshaped_arpegio.numpy()

# generate a more interesting chorale by adding some randomness
seed_arpegio = tf.reshape(shifted_tensor, [1, -1])
daringness = 10000

for _ in range(chorale_length):
    for _ in range(4):
        next_note_probas = chorale_model.predict(seed_arpegio, verbose=0)[0, -1:]
        rescaled_logits = tf.math.log(next_note_probas / daringness)
        next_note = tf.random.categorical(rescaled_logits, num_samples=1)
        seed_arpegio = tf.concat([seed_arpegio, next_note], axis=1)
shifted_arpegio = tf.where(seed_arpegio == 0, seed_arpegio, seed_arpegio + min_note - 1)
reshaped_arpegio = tf.reshape(shifted_arpegio, shape=[-1, 4])
generated_chorale_arr = reshaped_arpegio.numpy()
generated_chorale = generated_chorale_arr.tolist()

def convert_chorale(in_chorale, num_chords=None, save_path=None):
    if num_chords is not None:
        in_chorale = in_chorale[:num_chords]
    
    chorale_stream = stream.Score()
    chorale_part = stream.Part()
    chorale_stream.append(chorale_part)
    chorale_notes = [note.Note(chorale_note) for chord in in_chorale
                    for chorale_note in chord]
    chorale_part.append(chorale_notes)

    chorale_stream.append(meter.TimeSignature('4/4'))
    chorale_stream.append(tempo.MetronomeMark(number=160))
    
    if save_path is not None:
        midi_save_path=save_path
        chorale_stream.write(fmt='midi', fp=midi_save_path)
    
    return in_chorale

midi_save_path = "./outputs/generated_chorale.mid"
_ = convert_chorale(generated_chorale, save_path=midi_save_path)
play_chorale(midi_save_path)
