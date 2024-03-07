#%% from biological to artifical neurons
# the perceptron
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 0)

clf = Perceptron(random_state=42)
clf.fit(X, y)

X_new = [[2, 0.5], [3, 1]]
y_pred = clf.predict(X_new)

# regression MLPs
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X_train_val, X_test, y_train_val, y_test = train_test_split(
    housing.data, housing.target, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, random_state=42)

mlp_reg = MLPRegressor(hidden_layer_sizes=[50, 50, 50], random_state=42)
pipeline = make_pipeline(StandardScaler(), mlp_reg)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(rmse)

#%% implementing MLPs with keras
# building an image classifier using the sequential API
import tensorflow as tf
import tensorflow.keras as tfk
import pandas as pd
import matplotlib.pyplot as plt

fashion_mnist = tfk.datasets.fashion_mnist.load_data()
(X_train_val, y_train_val), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_val[:-5000], y_train_val[:-5000]
X_val, y_val = X_train_val[-5000:], y_train_val[-5000:]
print(X_train.shape)
print(X_train.dtype)

X_train, X_val, X_test = X_train / 255., X_val / 255., X_test / 255.
class_names = ['T-shirt/Top', 'Trousers', 'Sweater', 'Dress', 'Coat', 'Sandal', 'Shirt',
               'Sneakers', 'Bag', 'Boot']
print(class_names[y_train[0]])

tf.random.set_seed(42)
model = tfk.models.Sequential()
model.add(tfk.layers.Input(shape=[28, 28]))
model.add(tfk.layers.Flatten())
model.add(tfk.layers.Dense(300, activation='relu'))
model.add(tfk.layers.Dense(100, activation='relu'))
model.add(tfk.layers.Dense(10, activation='softmax'))

tfk.backend.clear_session()
model = tfk.models.Sequential([
    tfk.layers.Flatten(input_shape=[28, 28]),
    tfk.layers.Dense(300, activation='relu'),
    tfk.layers.Dense(100, activation='relu'),
    tfk.layers.Dense(10, activation='softmax')
])

model.summary()
model.layers
hidden1 = model.layers[1]
print(hidden1.name)
print(model.get_layer('dense') is hidden1)

weights, biases = hidden1.get_weights()
print(weights)
print(weights.shape)
print(biases)
print(biases.shape)

model.compile(loss='sparse_categorical_crossentropy', optimizer='SGD',
              metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val))

pd.DataFrame(history.history).plot(
    xlim=[0, 29], ylim=[0, 1], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])

model.evaluate(X_test, y_test)

X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba.round(2))

y_pred = np.argmax(y_proba, axis=-1)
print(y_pred)
print(np.array(class_names)[y_pred])
y_new = y_test[:3]
print(y_new)

# building a regression MLP using the sequential API
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housing = fetch_california_housing()
X_train_val, X_test, y_train_val, y_test = train_test_split(
    housing.data, housing.target, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,random_state=42)

tf.random.set_seed(42)
norm_layer = tfk.layers.Normalization(input_shape=X_train.shape[1:])
model = tfk.Sequential([
    norm_layer,
    tfk.layers.Dense(50, activation='relu'),
    tfk.layers.Dense(50, activation='relu'),
    tfk.layers.Dense(50, activation='relu'),
    tfk.layers.Dense(1)
    ])
optimiser = tfk.optimizers.Adam(learning_rate=1e-3)
model.compile(loss='mse', optimizer=optimiser, metrics=["RootMeanSquaredError"])
norm_layer.adapt(X_train)
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))
mse_test, rmse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
print(X_new)
y_pred = model.predict(X_new)
print(y_pred)

# building complex models using the functional API
tfk.backend.clear_session()
normalisation_layer = tfk.layers.Normalization()
hidden_layer1 = tfk.layers.Dense(30, activation='relu')
hidden_layer2 = tfk.layers.Dense(30, activation='relu')
concat_layer = tfk.layers.Concatenate()
output_layer = tfk.layers.Dense(1)

input_ = tfk.layers.Input(shape=X_train.shape[1:])
normalised = normalisation_layer(input_)
hidden1 = hidden_layer1(normalised)
hidden2 = hidden_layer2(hidden1)
concat = concat_layer([normalised, hidden2])
output = output_layer(concat)

model = tfk.Model(inputs=[input_], outputs=[output])


tfk.backend.clear_session()
input_wide = tfk.layers.Input(shape=[5])
input_deep = tfk.layers.Input(shape=[6])
norm_layer_wide = tfk.layers.Normalization()
norm_layer_deep = tfk.layers.Normalization()
norm_wide = norm_layer_wide(input_wide)
norm_deep = norm_layer_deep(input_deep)
hidden1 = tfk.layers.Dense(30, activation='relu')(norm_deep)
hidden2 = tfk.layers.Dense(30, activation='relu')(hidden1)
concat = tfk.layers.concatenate([norm_wide, hidden2])
output = tfk.layers.Dense(1)(concat)
model = tfk.Model(inputs=[input_wide, input_deep], outputs=[output])

optimiser = tfk.optimizers.Adam(learning_rate=1e-3)
model.compile(loss='mse', optimizer=optimiser, metrics=['RootMeanSquaredError'])

X_train_wide, X_train_deep = X_train[:, :5], X_train[:, 2:]
X_val_wide, X_val_deep = X_val[:, :5], X_val[:, 2:]
X_test_wide, X_test_deep = X_test[:, :5], X_test[:, 2:]
X_new_wide, X_new_deep = X_test_wide[:3], X_test_deep[:3]

norm_layer_wide.adapt(X_train_wide)
norm_layer_deep.adapt(X_train_deep)
history = model.fit((X_train_wide, X_train_deep), y_train, epochs=20,
                    validation_data=((X_val_wide, X_val_deep), y_val))
mse_test = model.evaluate((X_test_wide, X_test_deep), y_test)
y_pred = model.predict((X_new_wide, X_new_deep))
print(y_pred)


tfk.backend.clear_session()
input_wide = tfk.layers.Input(shape=[5])
input_deep = tfk.layers.Input(shape=[6])
norm_layer_wide = tfk.layers.Normalization()
norm_layer_deep = tfk.layers.Normalization()
norm_wide = norm_layer_wide(input_wide)
norm_deep = norm_layer_deep(input_deep)
hidden1 = tfk.layers.Dense(30, activation='relu')(norm_deep)
hidden2 = tfk.layers.Dense(30, activation='relu')(hidden1)
concat = tfk.layers.concatenate([norm_wide, hidden2])
output = tfk.layers.Dense(1)(concat)
aux_output = tfk.layers.Dense(1)(hidden2)
model = tfk.Model(inputs=[input_wide, input_deep], outputs=[output, aux_output])

optimiser = tfk.optimizers.Adam(learning_rate=1e-3)
model.compile(loss=('mse', 'mse'), loss_weights=(0.9, 0.1), optimizer=optimiser,
              metrics=['RootMeanSquaredError'])

norm_layer_wide.adapt(X_train_wide)
norm_layer_deep.adapt(X_train_deep)
history = model.fit(
    (X_train_wide, X_train_deep), (y_train, y_train), epochs=20,
    validation_data=((X_val_wide, X_val_deep), (y_val, y_val))
    )
eval_results = model.evaluate((X_test_wide, X_test_deep), (y_test, y_test))
weighted_sum_of_losses, main_loss, aux_loss, main_rmse, aux_rmse = eval_results

y_pred_main, y_pred_aux = model.predict((X_new_wide, X_new_deep))
y_pred_tuple = model.predict((X_new_wide, X_new_deep))
y_pred = dict(zip(model.output_names, y_pred_tuple))
print(y_pred)

# using the subclassing API to build dynamic models
class WideAndDeepModel(tfk.Model):
    def __init__(self, units=30, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.norm_layer_wide = tfk.layers.Normalization()
        self.norm_layer_deep = tfk.layers.Normalization()
        self.hidden1 = tfk.layers.Dense(units, activation=activation)
        self.hidden2 = tfk.layers.Dense(units, activation=activation)
        self.main_output = tfk.layers.Dense(1)
        self.aux_output = tfk.layers.Dense(1)

    def call(self, inputs):
        input_wide, input_deep = inputs
        norm_wide = self.norm_layer_wide(input_wide)
        norm_deep = self.norm_layer_deep(input_deep)
        hidden1 = self.hidden1(norm_deep)
        hidden2 = self.hidden2(hidden1)
        concat = tfk.layers.concatenate([norm_wide, hidden2])
        output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return output, aux_output

model = WideAndDeepModel(30, activation='relu', name='my_cool_model')

optimiser = tfk.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", loss_weights=[0.9, 0.1], optimizer=optimiser,
              metrics=["RootMeanSquaredError"])
model.norm_layer_wide.adapt(X_train_wide)
model.norm_layer_deep.adapt(X_train_deep)
history = model.fit(
    (X_train_wide, X_train_deep), (y_train, y_train), epochs=10,
    validation_data=((X_val_wide, X_val_deep), (y_val, y_val)))
eval_results = model.evaluate((X_test_wide, X_test_deep), (y_test, y_test))
weighted_sum_of_losses, main_loss, aux_loss, main_rmse, aux_rmse = eval_results
y_pred_main, y_pred_aux = model.predict((X_new_wide, X_new_deep))

# saving and restoring a model
model.save('./outputs/my_keras_model', save_format='tf')

model = tfk.models.load_model('./outputs/my_keras_model')
y_pred_main, y_pred_aux = model.predict((X_new_wide, X_new_deep))

# using callbacks
checkpoint_cb = tfk.callbacks.ModelCheckpoint('./outputs/my_checkpoints',
                                              save_weights_only=True)
history = model.fit((X_train_wide, X_train_deep), (y_train, y_train), epochs=10,
                    validation_data=((X_val_wide, X_val_deep), (y_val, y_val)),
                    callbacks=[checkpoint_cb])

early_stopping_cb = tfk.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit((X_train_wide, X_train_deep), (y_train, y_train), epochs=10,
                    validation_data=((X_val_wide, X_val_deep), (y_val, y_val)),
                    callbacks=[checkpoint_cb, early_stopping_cb])

class PrintValTrainRatioCallback(tfk.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        ratio = logs["val_loss"] / logs["loss"]
        print(f"Epoch={epoch}, val/train={ratio:.2f}")

# %% using tensorboard for visualisation
from pathlib import Path
from time import strftime

def get_run_logdir(root_logdir="./logs/my_logs"):
    return Path(root_logdir) / strftime("run_%Y_%m_%d-%H_%M_%S")

run_logdir = get_run_logdir()

tensorboard_cb = tfk.callbacks.TensorBoard(run_logdir, profile_batch=(100, 200))
history = model.fit((X_train_wide, X_train_deep), (y_train, y_train), epochs=10,
                    validation_data=((X_val_wide, X_val_deep), (y_val, y_val)),
                    callbacks=[tensorboard_cb])

test_logdir = get_run_logdir()
writer = tf.summary.create_file_writer(str(test_logdir))
with writer.as_default():
    for step in range(1, 1000 + 1):
        tf.summary.scalar("my_scalar", np.sin(step / 10), step=step)
        
        data = (np.random.randn(100) + 2) * step / 100
        tf.summary.histogram("my_hist", data, buckets=50, step=step)
        
        images = np.random.randn(2, 32, 32, 3)
        tf.summary.image("my_images", images * step / 1000, step=step)
        
        texts = ["The step is ", str(step), "It'\s square is " + str(step**2)]
        tf.summary.text("my_text", texts, step=step)
        
        sine_wave = tf.math.sin(tf.range(12000) / 48000 * 2 * np.pi * step)
        audio = tf.reshape(tf.cast(sine_wave, tf.float32), [1, -1, 1])
        tf.summary.audio("my_audio", audio, sample_rate=48000, step=step)

# %% fine-tuning neural network hyperparameters
import keras_tuner as kt

fashion_mnist = tfk.datasets.fashion_mnist.load_data()
(X_train_val, y_train_val), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_val[:-5000], y_train_val[:-5000]
X_val, y_val = X_train_val[-5000:], y_train_val[-5000:]

X_train, X_val, X_test = X_train / 255., X_val / 255., X_test / 255.
class_names = ['T-shirt/Top', 'Trousers', 'Sweater', 'Dress', 'Coat', 'Sandal', 'Shirt',
               'Sneakers', 'Bag', 'Boot']

def build_model(hp):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=8, default=2)
    n_neurons = hp.Int("n_neurons", min_value=16, max_value=256)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2,
                             sampling="log")
    optimiser = hp.Choice("optimiser", values=["SGD", "Adam"])
    if optimiser == "SGD":
        optimiser = tfk.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimiser = tfk.optimizers.Adam(learning_rate=learning_rate)
    
    model = tfk.Sequential()
    model.add(tfk.layers.Flatten())
    for _ in range(n_hidden):
        model.add(tfk.layers.Dense(n_neurons, activation="relu"))
    model.add(tfk.layers.Dense(10, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimiser,
                  metrics="accuracy")
    return model

rnd_search_tuner = kt.RandomSearch(
    build_model, objective="val_accuracy", max_trials=5, overwrite=True,
    directory="./outputs/my_fashion_mnist", project_name="my_rnd_search", seed=42)
rnd_search_tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

top3_models = rnd_search_tuner.get_best_models(num_models=3)
best_model = top3_models[0]

top3_params = rnd_search_tuner.get_best_hyperparameters(num_trials=3)
print(top3_params[0].values)

best_trial = rnd_search_tuner.oracle.get_best_trials(num_trials=1)[0]
print(best_trial.summary())
print(best_trial.metrics.get_last_value("val_accuracy"))

best_model.fit(X_train_val, y_train_val, epochs=10)
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print(test_accuracy)

class MyClassificationHyperModel(kt.HyperModel):
    def build(self, hp):
        return build_model(hp)
    
    def fit(self, hp, model, X, y, **kwargs):
        if hp.Boolean("normalise"):
            norm_layer = tfk.layers.Normalization()
            X = norm_layer(X)
        return model.fit(X, y, **kwargs)
    
hyperband_tuner = kt.Hyperband(MyClassificationHyperModel(), objective="val_accuracy",
                               seed=42, max_epochs=10, factor=3, hyperband_iterations=2,
                               overwrite=True, directory="./outputs/my_fashion_mnist",
                               project_name="hyperband")

root_logdir = Path(hyperband_tuner.project_dir) / "tensorboard"
tensorboard_cb = tfk.callbacks.TensorBoard(root_logdir)
early_stopping_cb = tfk.callbacks.EarlyStopping(patience=2)
hyperband_tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val),
                       callbacks=[early_stopping_cb, tensorboard_cb])

bayesian_opt_tuner = kt.BayesianOptimization(
    MyClassificationHyperModel(), objective="val_accuracy", seed=42, max_trials=10,
    alpha=1e-4, beta=2.6, overwrite=True, directory="./outputs/my_fashion_mnist",
    project_name="bayesian_opt")
bayesian_opt_tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val),
                          callbacks=[early_stopping_cb])

#%% Coding Exercises: Exercise 10
import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import keras_tuner as kt
# train a deep MLP on the MNIST dataset
# determine the optimal learning rate by growing it exponentially, plotting the loss and
# finding where the loss shoots up
(X_train_val, y_train_val), (X_test, y_test) = tfk.datasets.mnist.load_data()

X_val, X_train = X_train_val[:5000] / 255., X_train_val[5000:] / 255.
y_val, y_train = y_train_val[:5000], y_train_val[5000:]
X_test = X_test / 255.

class ExponentialLearningRate(tfk.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []
    def on_batch_end(self, batch, logs):
        self.rates.append(tfk.backend.get_value(self.model.optimizer.learning_rate))
        self.losses.append(logs["loss"])
        tfk.backend.set_value(self.model.optimizer.learning_rate,
                              self.model.optimizer.learning_rate * self.factor)

np.random.seed(42)
tf.random.set_seed(42)

model = tfk.models.Sequential([
    tfk.layers.Flatten(input_shape=[28, 28]),
    tfk.layers.Dense(300, activation='relu'),
    tfk.layers.Dense(100, activation='relu'),
    tfk.layers.Dense(10, activation='softmax')
])
optimiser=tfk.optimizers.SGD(learning_rate=1e-3)
model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
expon_lr = ExponentialLearningRate(factor=1.005)

history = model.fit(X_train, y_train, epochs=1, validation_data=(X_val, y_val),
                    callbacks=[expon_lr])

plt.plot(expon_lr.rates, expon_lr.losses)
plt.gca().set_xscale("log")
plt.hlines(min(expon_lr.losses), min(expon_lr.rates), max(expon_lr.rates))
plt.axis([min(expon_lr.rates), max(expon_lr.rates), 0, expon_lr.losses[0]])
plt.xlabel("Learning rate")
plt.ylabel("Loss")


tfk.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = tfk.models.Sequential([
    tfk.layers.Flatten(input_shape=[28, 28]),
    tfk.layers.Dense(300, activation='relu'),
    tfk.layers.Dense(100, activation='relu'),
    tfk.layers.Dense(10, activation='softmax')
])
optimiser=tfk.optimizers.SGD(learning_rate=3e-1)
model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

run_idx = 1
run_logdir = Path() / "./logs/ch10_ex10_run_{:03d}".format(run_idx)
print(run_logdir)

early_stopping_cb = tfk.callbacks.EarlyStopping(patience=20)
checkpoint_cb = tfk.callbacks.ModelCheckpoint("./outputs/ch10_ex10_model",
                                              save_best_only=True)
tensorboard_cb = tfk.callbacks.TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val),
                    callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb])

model = tfk.models.load_model("./outputs/ch10_ex10_model")
model.evaluate(X_test, y_test)

# tune the hyperparameters using Keras Tuner with all the bells and whistles - save
# checkpoints, use early stopping and plot learning curves with Tensorboard
def build_model(hp):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=8, default=2)
    n_neurons = hp.Int("n_neurons", min_value=16, max_value=256)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1,
                             sampling="log")
    optimiser = hp.Choice("optimiser", values=["SGD", "Adam"])
    if optimiser == "SGD":
        optimiser = tfk.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimiser = tfk.optimizers.Adam(learning_rate=learning_rate)
    
    model = tfk.Sequential()
    model.add(tfk.layers.Flatten())
    for _ in range(n_hidden):
        model.add(tfk.layers.Dense(n_neurons, activation="relu"))
    model.add(tfk.layers.Dense(10, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimiser,
                  metrics="accuracy")
    return model

tfk.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

rnd_search_tuner = kt.RandomSearch(
    build_model, objective="val_accuracy", max_trials=15, overwrite=True,
    directory="./outputs/ch10_ex10_tuned_model", project_name="rnd_search", seed=42)

root_logdir = Path(rnd_search_tuner.project_dir) / "tensorboard"
tensorboard_cb = tfk.callbacks.TensorBoard(root_logdir)
early_stopping_cb = tfk.callbacks.EarlyStopping(patience=10)
checkpoint_cb = tfk.callbacks.ModelCheckpoint(
    "./outputs/ch10_ex10_tuned_model/best_model", save_best_only=True)

rnd_search_tuner.search(X_train, y_train, epochs=100, validation_data=(X_val, y_val),
                        callbacks=[early_stopping_cb, checkpoint_cb, tensorboard_cb])

top3_models = rnd_search_tuner.get_best_models(num_models=3)
best_model = top3_models[0]

top3_params = rnd_search_tuner.get_best_hyperparameters(num_trials=3)
print(top3_params[0].values)

best_trial = rnd_search_tuner.oracle.get_best_trials(num_trials=1)[0]
print(best_trial.summary())
print(best_trial.metrics.get_last_value("val_accuracy"))

best_model.fit(X_train_val, y_train_val, epochs=20)
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print(test_accuracy)
