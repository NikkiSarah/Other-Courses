#%% the vanishing/exploding gradients problem
import tensorflow as tf
import tensorflow.keras as tfk

dense = tfk.layers.Dense(50, activation="relu")

he_avg_init = tfk.initializers.VarianceScaling(scale=2., mode="fan_avg",
                                               distribution="uniform")
dense = tfk.layers.Dense(50, activation="sigmoid", kernel_initializer=he_avg_init)

leaky_relu = tfk.layers.LeakyReLU(alpha=0.2)
dense = tfk.layers.Dense(50, activation=leaky_relu, kernel_initializer="he_normal")

model = tfk.models.Sequential([
    # [...]
    tfk.layers.Dense(50, kernel_initializer="he_normal"),
    tfk.layers.LeakyReLU(alpha=0.2)
    # [...]
    ])

# batch normalisation
model = tfk.models.Sequential([
    tfk.layers.Flatten(input_shape=[28, 28]),
    tfk.layers.BatchNormalization(),
    tfk.layers.Dense(300, activation="relu", kernel_initializer='he_normal'),
    tfk.layers.BatchNormalization(),
    tfk.layers.Dense(100, activation="relu", kernel_initializer='he_normal'),
    tfk.layers.BatchNormalization(),
    tfk.layers.Dense(10, activation='softmax')
])
model.summary()

[(var.name, var.trainable) for var in model.layers[1].variables]

tfk.backend.clear_session()
model = tfk.models.Sequential([
    tfk.layers.Flatten(input_shape=[28, 28]),
    tfk.layers.Dense(300, kernel_initializer='he_normal', use_bias=False),
    tfk.layers.BatchNormalization(),
    tfk.layers.Activation("relu"),
    tfk.layers.Dense(100, kernel_initializer='he_normal', use_bias=False),
    tfk.layers.BatchNormalization(),
    tfk.layers.Activation("relu"),
    tfk.layers.Dense(10, activation='softmax')
])
model.summary()

# gradient clipping
optimiser = tfk.optimizers.SGD(clipvalue=1.0)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser)

#%% reusing pretrained layers
import numpy as np

# transfer learning with keras
fashion_mnist = tfk.datasets.fashion_mnist.load_data()
(X_train_val, y_train_val), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_val[:-5000], y_train_val[:-5000]
X_val, y_val = X_train_val[-5000:], y_train_val[-5000:]
X_train, X_val, X_test = X_train / 255., X_val / 255., X_test / 255.
class_names = ['T-shirt/Top', 'Trousers', 'Sweater', 'Dress', 'Coat', 'Sandal', 'Shirt',
               'Sneakers', 'Bag', 'Boot']

pos_class_id = class_names.index("Sweater")
neg_class_id = class_names.index("T-shirt/Top")

def split_dataset(X, y):
    y_for_B = (y == pos_class_id) | (y == neg_class_id)
    y_A = y[~y_for_B]
    y_B = (y[y_for_B] == pos_class_id).astype(np.float32)
    old_class_ids = list(set(range(10)) - set([neg_class_id, pos_class_id]))
    for old_class_id, new_class_id in zip(old_class_ids, range(8)):
        y_A[y_A == old_class_id] = new_class_id
    return ((X[~y_for_B], y_A), (X[y_for_B], y_B))

(X_train_A, y_train_A), (X_train_B, y_train_B) = split_dataset(X_train, y_train)
(X_val_A, y_val_A), (X_val_B, y_val_B) = split_dataset(X_val, y_val)
(X_test_A, y_test_A), (X_test_B, y_test_B) = split_dataset(X_test, y_test)
X_train_B = X_train_B[:200]
y_train_B = y_train_B[:200]

tfk.backend.clear_session()
tf.random.set_seed(42)
model_A = tfk.Sequential([
    tfk.layers.Flatten(input_shape=[28, 28]),
    tfk.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
    tfk.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
    tfk.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
    tfk.layers.Dense(8, activation="softmax")
])

model_A.compile(loss="sparse_categorical_crossentropy",
                optimizer=tfk.optimizers.SGD(learning_rate=0.001),
                metrics=["accuracy"])
history = model_A.fit(X_train_A, y_train_A, epochs=20, validation_data=(X_val_A, y_val_A))
model_A.save("./outputs/my_model_A")

model_A = tfk.models.load_model("./outputs/my_model_A")
model_B_on_A = tfk.Sequential(model_A.layers[:-1])
model_B_on_A.add(tfk.layers.Dense(1, activation="sigmoid"))

model_A_clone = tfk.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False

optimiser = tfk.optimizers.SGD(learning_rate=0.001)
model_B_on_A.compile(loss='binary_crossentropy', optimizer=optimiser,
                     metrics=['accuracy'])
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4,
                           validation_data=(X_val_B, y_val_B))

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True

optimiser = tfk.optimizers.SGD(learning_rate=0.001)
model_B_on_A.compile(loss='binary_crossentropy', optimizer=optimiser,
                     metrics=['accuracy'])
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16, 
                           validation_data=(X_val_B, y_val_B))

model_B_on_A.evaluate(X_test_B, y_test_B)

#%% faster optimisers
import math

# momentum
optimiser = tfk.optimizers.SGD(learning_rate=1e-3, momentum=0.9)
# nesterov accelerated gradient
optimiser = tfk.optimizers.SGD(learning_rate=1e-3, momentum=0.9, nesterov=True)
# rmsprop
optimiser = tfk.optimizers.RMSprop(learning_rate=1e-3, rho=0.9)
# adam and nadam
optimiser = tfk.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999)

# learning rate scheduling: power
optimiser = tfk.optimizers.legacy.SGD(learning_rate=0.01, decay=1e-4)

# learning rate scheduling: exponential and piecewise
def exponential_decay_fn(epoch):
    return 0.01 * 0.1**(epoch / 20)


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn

exponential_decay_fn = exponential_decay(lr0=0.01, s=20)

def build_model(seed=42):
    tf.random.set_seed(seed)
    return tfk.Sequential([
        tfk.layers.Flatten(input_shape=[28, 28]),
        tfk.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
        tfk.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
        tfk.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
        tfk.layers.Dense(10, activation="softmax")
    ])

tf.random.set_seed(42)
model = build_model()
optimiser = tfk.optimizers.SGD(learning_rate=0.001)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimiser,
              metrics=["accuracy"])
lr_scheduler = tfk.callbacks.LearningRateScheduler(exponential_decay_fn)
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val),
                    callbacks=[lr_scheduler])

def exponential_decay_fn(epoch, lr):
    return lr * 0.1**(1 / 20)


def piecewise_constant_fn(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 15:
        return 0.005
    else:
        return 0.001
    
# learning rate scheduling: performance
model = build_model()
optimiser = tfk.optimizers.SGD(learning_rate=0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimiser,
              metrics=["accuracy"])

lr_scheduler = tfk.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val),
                    callbacks=[lr_scheduler])

batch_size = 32
n_epochs = 25
n_steps = n_epochs * math.ceil(len(X_train) / batch_size)
scheduled_learning_rate = tfk.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01, decay_steps=n_steps, decay_rate=0.1)
optimiser = tfk.optimizers.SGD(learning_rate=scheduled_learning_rate)

#%% avoiding overfitting through regularisation
from functools import partial

# l1 and l2 regularisation
layer = tfk.layers.Dense(100, activation='relu', kernel_initializer='he_normal',
                         kernel_regularizer=tfk.regularizers.l2(0.01))

RegularisedDense = partial(tfk.layers.Dense, activation='relu',
                           kernel_initializer='he_normal',
                           kernel_regularizer=tfk.regularizers.l2(0.01))

tfk.backend.clear_session()
model = tfk.Sequential([
    tfk.layers.Flatten(input_shape=[28, 28]),
    RegularisedDense(300),
    RegularisedDense(100),
    RegularisedDense(10, activation='softmax')
    ])
                         
# dropout
tfk.backend.clear_session()
model = tfk.Sequential([
    tfk.layers.Flatten(input_shape=[28, 28]),
    tfk.layers.Dropout(rate=0.2),
    tfk.layers.Dense(100, activation='relu', kernel_initializer='he_normal'),
    tfk.layers.Dropout(rate=0.2),
    tfk.layers.Dense(100, activation='relu', kernel_initializer='he_normal'),
    tfk.layers.Dropout(rate=0.2),
    tfk.layers.Dense(10, activation='softmax')
])
optimiser = tfk.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimiser,
              metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# monte carlo dropout
y_probas = np.stack([model(X_test, training=True) for sample in range(100)])
y_proba = y_probas.mean(axis=0)

print(model.predict(X_test[:1]).round(3))
print(y_proba[0].round(3))

y_std = y_probas.std(axis=0)
print(y_std[0].round(3))

y_pred = y_proba.argmax(axis=1)
accuracy = (y_pred == y_test).sum() / len(y_test)
print(accuracy)

class MCDropout(tfk.layers.Dropout):
    def call(self, inputs, training=False):
        return super().call(inputs, training=True)

# max-norm regularisation
dense = tfk.layers.Dense(100, activation='relu', kernel_initializer='he_normal',
                         kernel_constraint=tfk.constraints.max_norm(1.))

#%% Coding Exercises: Exercise 8
import tensorflow as tf
import tensorflow.keras as tfk
import os
import numpy as np
import math
import matplotlib.pyplot as plt

# build a DNN with 20 hidden layers of 100 neurons each. Use He initialisation and the
# swish activation function
# use nadam optimisation and early stopping to train the network
(X_train_val, y_train_val), (X_test, y_test) = tfk.datasets.cifar10.load_data()

X_train, X_val = X_train_val[:35000] / 255.0, X_train_val[35000:] / 255.0
y_train, y_val= y_train_val[:35000], y_train_val[35000:]
X_test = X_test / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck']

def get_run_logdir(run_idx, learning_rate):
    run_id = f"run{run_idx}_lr{learning_rate}"
    return os.path.join(root_logdir, run_id)

root_logdir = '.\logs\ch11_ex8\\base'

idx = 0
for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    tfk.backend.clear_session()
    model = tfk.Sequential()
    model.add(tfk.layers.Flatten(input_shape=[32, 32, 3]))
    for _ in range(20):
        model.add(tfk.layers.Dense(100, activation="swish",
                                   kernel_initializer="he_normal"))
    model.add(tfk.layers.Dense(10, activation="softmax"))

    optimiser = tfk.optimizers.experimental.Nadam(learning_rate=lr)
    
    run_logdir = get_run_logdir(idx, lr)
    idx += 1
    print(run_logdir)
    
    tensorboard_cb = tfk.callbacks.TensorBoard(run_logdir)
    earlystopping_cb = tfk.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    callback_list = [tensorboard_cb, earlystopping_cb]
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser,
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, callbacks=callback_list,
              validation_data=(X_val, y_val))
# the best model had a learning rate of 1e-4. It was able to achieve a validation accuracy
# of 0.4803 after 20 epochs and took 3.51 minutes to train.

# add batch normalisation
root_logdir = '.\logs\ch11_ex8\\bn'

idx = 0
for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    tfk.backend.clear_session()
    model = tfk.Sequential()
    model.add(tfk.layers.Flatten(input_shape=[32, 32, 3]))
    for _ in range(20):
        model.add(tfk.layers.Dense(100, kernel_initializer="he_normal"))
        model.add(tfk.layers.BatchNormalization())
        model.add(tfk.layers.Activation("swish"))
    model.add(tfk.layers.Dense(10, activation="softmax"))

    optimiser = tfk.optimizers.experimental.Nadam(learning_rate=lr)
    
    run_logdir = get_run_logdir(idx, lr)
    idx += 1
    print(run_logdir)
    
    tensorboard_cb = tfk.callbacks.TensorBoard(run_logdir)
    earlystopping_cb = tfk.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    callback_list = [tensorboard_cb, earlystopping_cb]
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser,
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, callbacks=callback_list,
              validation_data=(X_val, y_val))
# the best model had a learning rate of 1e-3. It was able to achieve a validation accuracy
# of 0.4753 after 20 epochs and took 9.34 minutes to train.

# replace batch normalisation with SELU and make the necessary adjustments to ensure the
# network self-normalises (i.e. standardise the input features, use LeCun normal
# initialisation, ensure the DNN only includes dense layers etc)
X_means = X_train.mean(axis=0)
X_stds = X_train.std(axis=0)
X_train_scaled = (X_train - X_means) / X_stds
X_val_scaled = (X_val - X_means) / X_stds
X_test_scaled = (X_test - X_means) / X_stds

root_logdir = '.\logs\ch11_ex8\\selu'

idx = 0
for lr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    tfk.backend.clear_session()
    model = tfk.Sequential()
    model.add(tfk.layers.Flatten(input_shape=[32, 32, 3]))
    for _ in range(20):
        model.add(tfk.layers.Dense(100, activation="selu",
                                   kernel_initializer="lecun_normal"))
    model.add(tfk.layers.Dense(10, activation="softmax"))

    optimiser = tfk.optimizers.experimental.Nadam(learning_rate=lr)
    
    run_logdir = get_run_logdir(idx, lr)
    idx += 1
    print(run_logdir)
    
    tensorboard_cb = tfk.callbacks.TensorBoard(run_logdir)
    earlystopping_cb = tfk.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    callback_list = [tensorboard_cb, earlystopping_cb]
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser,
                  metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=20, callbacks=callback_list,
              validation_data=(X_val_scaled, y_val))
# the best model had a learning rate of 1e-3. It was able to achieve a validation accuracy
# of 0.4877 after 20 epochs and took 3.17 minutes to train.

# regularise the model with alpha dropout
root_logdir = '.\logs\ch11_ex8\\dropout'

idx = 0
for lr in [1e-3, 1e-4, 1e-5, 1e-6]:
    for dr in [0.1, 0.2, 0.3, 0.4, 0.5]:
        tfk.backend.clear_session()
        model = tfk.Sequential()
        model.add(tfk.layers.Flatten(input_shape=[32, 32, 3]))
        for _ in range(20):
            model.add(tfk.layers.Dense(100, activation="selu",
                                       kernel_initializer="lecun_normal"))
        model.add(tfk.layers.AlphaDropout(rate=dr))
        model.add(tfk.layers.Dense(10, activation="softmax"))
    
        optimiser = tfk.optimizers.experimental.Nadam(learning_rate=lr)
        
        run_logdir = get_run_logdir(idx, lr)
        idx += 1
        print(run_logdir)
        
        tensorboard_cb = tfk.callbacks.TensorBoard(run_logdir)
        earlystopping_cb = tfk.callbacks.EarlyStopping(patience=10,
                                                       restore_best_weights=True)
        callback_list = [tensorboard_cb, earlystopping_cb]
        
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser,
                      metrics=['accuracy'])
        model.fit(X_train_scaled, y_train, epochs=20, callbacks=callback_list,
                  validation_data=(X_val_scaled, y_val))
# the best model had a learning rate of 1e-3 and dropout of 0.1. It was able to achieve a
# validation accuracy of 0.4771 after 20 epochs and took 5.83 minutes to train.


# without retraining the model, see if there is better performance with MC Dropout
tfk.backend.clear_session()
model = tfk.Sequential()
model.add(tfk.layers.Flatten(input_shape=[32, 32, 3]))
for _ in range(20):
    model.add(tfk.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"))
model.add(tfk.layers.AlphaDropout(rate=0.1))
model.add(tfk.layers.Dense(10, activation="softmax"))

optimiser = tfk.optimizers.experimental.Nadam(learning_rate=1e-3)

earlystopping_cb = tfk.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser,
              metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=20, callbacks=earlystopping_cb,
          validation_data=(X_val_scaled, y_val))
model.evaluate(X_val_scaled, y_val)

class MCAlphaDropout(tfk.layers.AlphaDropout):
    def call(self, inputs):
        return super().call(inputs, training=True)
    
mc_model = tfk.Sequential([
    (
        MCAlphaDropout(layer.rate)
        if isinstance(layer, tfk.layers.AlphaDropout)
        else layer
    )
    for layer in model.layers
])

def mc_dropout_predict_probas(mc_model, X, n_samples=10):
    Y_probas = [mc_model.predict(X) for sample in range(n_samples)]
    return np.mean(Y_probas, axis=0)

def mc_dropout_predict_classes(mc_model, X, n_samples=10):
    Y_probas = mc_dropout_predict_probas(mc_model, X, n_samples)
    return Y_probas.argmax(axis=1)

tf.random.set_seed(42)
y_pred = mc_dropout_predict_classes(mc_model, X_val_scaled)
accuracy = (y_pred == y_val[:, 0]).mean()
print(accuracy)

# retrain the model with 1cycle scheduling
tf.random.set_seed(42)

tfk.backend.clear_session()
model = tfk.Sequential()
model.add(tfk.layers.Flatten(input_shape=[32, 32, 3]))
for _ in range(20):
    model.add(tfk.layers.Dense(100, activation="selu",
                               kernel_initializer="lecun_normal"))
model.add(tfk.layers.AlphaDropout(rate=0.1))
model.add(tfk.layers.Dense(10, activation="softmax"))

optimiser = tfk.optimizers.experimental.Nadam()
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser,
              metrics=['accuracy'])

class ExponentialLearningRate(tfk.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []

    def on_epoch_begin(self, epoch, logs=None):
        self.sum_of_epoch_losses = 0

    def on_batch_end(self, batch, logs=None):
        mean_epoch_loss = logs["loss"]  # the epoch's mean loss so far 
        new_sum_of_epoch_losses = mean_epoch_loss * (batch + 1)
        batch_loss = new_sum_of_epoch_losses - self.sum_of_epoch_losses
        self.sum_of_epoch_losses = new_sum_of_epoch_losses
        self.rates.append(tfk.backend.get_value(self.model.optimizer.learning_rate))
        self.losses.append(batch_loss)
        tfk.backend.set_value(self.model.optimizer.learning_rate,
                    self.model.optimizer.learning_rate * self.factor)

def find_learning_rate(model, X, y, epochs=1, batch_size=32, min_rate=1e-4, max_rate=1):
    init_weights = model.get_weights()
    iterations = math.ceil(len(X) / batch_size) * epochs
    factor = (max_rate / min_rate) ** (1 / iterations)
    init_lr = tfk.backend.get_value(model.optimizer.learning_rate)
    tfk.backend.set_value(model.optimizer.learning_rate, min_rate)
    exp_lr = ExponentialLearningRate(factor)
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[exp_lr])
    tfk.backend.set_value(model.optimizer.learning_rate, init_lr)
    model.set_weights(init_weights)
    return exp_lr.rates, exp_lr.losses

def plot_lr_vs_loss(rates, losses):
    plt.plot(rates, losses, "b")
    plt.gca().set_xscale('log')
    max_loss = losses[0] + min(losses)
    plt.hlines(min(losses), min(rates), max(rates), color="k")
    plt.axis([min(rates), max(rates), 0, max_loss])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
        
rates, losses = find_learning_rate(model, X_train_scaled, y_train, epochs=1,
                                   batch_size=128)

plot_lr_vs_loss(rates, losses)

tfk.backend.clear_session()
model = tfk.Sequential()
model.add(tfk.layers.Flatten(input_shape=[32, 32, 3]))
for _ in range(20):
    model.add(tfk.layers.Dense(100, activation="selu",
                               kernel_initializer="lecun_normal"))
model.add(tfk.layers.AlphaDropout(rate=0.1))
model.add(tfk.layers.Dense(10, activation="softmax"))

optimiser = tfk.optimizers.experimental.Nadam(learning_rate=2e-2)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimiser,
              metrics=['accuracy'])

class OneCycleScheduler(tfk.callbacks.Callback):
    def __init__(self, iterations, max_lr=1e-3, start_lr=None,
                 last_iterations=None, last_lr=None):
        self.iterations = iterations
        self.max_lr = max_lr
        self.start_lr = start_lr or max_lr / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_lr = last_lr or self.start_lr / 1000
        self.iteration = 0

    def _interpolate(self, iter1, iter2, lr1, lr2):
        return (lr2 - lr1) * (self.iteration - iter1) / (iter2 - iter1) + lr1

    def on_batch_begin(self, batch, logs):
        if self.iteration < self.half_iteration:
            lr = self._interpolate(0, self.half_iteration, self.start_lr, self.max_lr)
        elif self.iteration < 2 * self.half_iteration:
            lr = self._interpolate(self.half_iteration, 2 * self.half_iteration,
                                   self.max_lr, self.start_lr)
        else:
            lr = self._interpolate(2 * self.half_iteration, self.iterations,
                                   self.start_lr, self.last_lr)
        self.iteration += 1
        tfk.backend.set_value(self.model.optimizer.learning_rate, lr)

n_epochs = 20
batch_size = 128
n_iterations = math.ceil(len(X_train_scaled) / batch_size) * n_epochs
onecycle = OneCycleScheduler(n_iterations, max_lr=0.05)
history = model.fit(X_train_scaled, y_train, epochs=n_epochs, batch_size=batch_size,
                    validation_data=(X_val_scaled, y_val), callbacks=[onecycle])
