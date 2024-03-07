#%% generating shakespearean text using a character RNN
import tensorflow.keras as tfk
import tensorflow as tf

# creating the training dataset
shakespeare_url = "https://homl.info/shakespeare"
filepath = tfk.utils.get_file("shakespeare.txt", shakespeare_url)

with open(filepath) as f:
    shakespeare_text = f.read()
print(shakespeare_text[:80])

text_vec_layer = tfk.layers.TextVectorization(split="character", standardize="lower")
text_vec_layer.adapt([shakespeare_text])
encoded = text_vec_layer([shakespeare_text])[0]
print(encoded)

encoded -= 2
num_tokens = text_vec_layer.vocabulary_size() - 2
print(num_tokens)
dataset_size = len(encoded)
print(dataset_size)

def to_dataset(sequence, length, shuffle=False, seed=None, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window_ds: window_ds.batch(length + 1))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=100000, seed=seed)
    
    ds = ds.batch(batch_size)   
    mapped_ds = ds.map(lambda window: (window[:, :-1], window[:, 1:]))
    
    return mapped_ds.prefetch(1)

length = 100
tf.random.set_seed(42)
train_set = to_dataset(encoded[:1000000], length=length, shuffle=True, seed=42)
val_set = to_dataset(encoded[1000000:1060000], length=length)
test_set = to_dataset(encoded[1060000:], length=length)

# building and training the char-rnn model
model = tfk.Sequential([
    tfk.layers.Embedding(input_dim=num_tokens, output_dim=16),
    tfk.layers.GRU(128, return_sequences=True),
    tfk.layers.Dense(num_tokens, activation="softmax")
    ])

model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
model_cb = tfk.callbacks.ModelCheckpoint("./outputs/my_shakepeare_model",
                                         monitor="val_accuracy", save_best_only=True)
history = model.fit(train_set, validation_data=val_set, epochs=10, callbacks=[model_cb])

shakespeare_model = tfk.Sequential([
    text_vec_layer,
    tfk.layers.Lambda(lambda X: X - 2),
    model
    ])

y_proba = shakespeare_model.predict(["To be or not to be"])[0, -1]
y_pred = tf.argmax(y_proba)
text_vec_layer.get_vocabulary()[y_pred + 2]

# generating fake shakespearean text
log_probas = tf.math.log([[0.5, 0.4, 0.1]])
tf.random.set_seed(42)
tf.random.categorical(log_probas, num_samples=8)

def next_char(text, temperature=1):
    y_proba = shakespeare_model.predict([text])[0, -1:]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0, 0]
    next_char = text_vec_layer.get_vocabulary()[char_id + 2]
    
    return next_char


def extend_text(text, num_chars=50, temperature=1):
    for _ in range(num_chars):
        text += next_char(text, temperature)
    return text

tf.random.set_seed(42)
print(extend_text("To be or not to be", temperature=0.01))
print(extend_text("To be or not to be", temperature=1))
print(extend_text("To be or not to be", temperature=100))

# stateful RNN
def to_dataset_for_stateful_rnn(sequence, length):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=length, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(length + 1)).batch(1)
    mapped_ds = ds.map(lambda window: (window[:, :-1], window[:, 1:]))
    
    return mapped_ds.prefetch(1)

stateful_train_set = to_dataset(encoded[:1000000], length)
stateful_val_set = to_dataset(encoded[1000000:1060000], length)
stateful_test_set = to_dataset(encoded[1060000:], length)

model = tfk.Sequential([
    tfk.layers.Embedding(input_dim=num_tokens, output_dim=16,
                         batch_input_shape=[1, None]),
    tfk.layers.GRU(128, return_sequences=True, stateful=True),
    tfk.layers.Dense(num_tokens, activation="softmax")
    ])
    
class ResetStatesCallback(tfk.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()

model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
history = model.fit(stateful_train_set, validation_data=stateful_val_set, epochs=10,
                    callbacks=[ResetStatesCallback(), model_cb])

#%% sentiment analysis
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import tensorflow.keras as tfk
import os
import tensorflow_hub as hub

raw_train_set, raw_val_set, raw_test_set = tfds.load(
    name="imdb_reviews",
    split=["train[:90%]", "train[90%:]", "test"],
    as_supervised=True
    )

tf.random.set_seed(42)
train_set = raw_train_set.shuffle(5000, seed=42).batch(32).prefetch(1)
val_set = raw_val_set.batch(32).prefetch(1)
test_set = raw_test_set.batch(32).prefetch(1)

for review, label in raw_train_set.take(4):
    print(review.numpy().decode("utf-8"))
    print("Label:", label.numpy())

vocab_size = 1000
text_vec_layer = tfk.layers.TextVectorization(max_tokens=vocab_size)
text_vec_layer.adapt(train_set.map(lambda reviews, labels: reviews))

embed_size = 128
tf.random.set_seed(42)
model = tfk.Sequential([
    text_vec_layer,
    tfk.layers.Embedding(vocab_size, embed_size),
    tfk.layers.GRU(128),
    tfk.layers.Dense(1, activation="sigmoid")
    ])

model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])
model.fit(train_set, validation_data=val_set, epochs=2)

# masking
inputs = tfk.layers.Input(shape=[], dtype=tf.string)
token_ids = text_vec_layer(inputs)
mask = tf.math.not_equal(token_ids, 0)
Z = tfk.layers.Embedding(vocab_size, embed_size)(token_ids)
Z = tfk.layers.GRU(128, dropout=0.2)(Z, mask=mask)
outputs =tfk.layers.Dense(1, activation="sigmoid")(Z)
model = tfk.Model(inputs=[inputs], outputs=[outputs])

text_vec_layer_ragged = tfk.layers.TextVectorization(max_tokens=vocab_size, ragged=True)
text_vec_layer_ragged.adapt(train_set.map(lambda reviews, labels: reviews))
print(text_vec_layer_ragged(["Great movie!", "This is DiCaprio's best role."]))

print(text_vec_layer(["Great movie!", "This is DiCaprio's best role."]))

# reusing pre-trained embeddings and language models
os.environ["TFHUB_CACHE_DIR"] = "./datasets/my_tfhub_cache"
model = tfk.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                   trainable=True,
                   dtype=tf.string,
                   input_shape=[]),
    tfk.layers.Dense(64, activation="relu"),
    tfk.layers.Dense(1, activation="sigmoid")
    ])
model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])
model.fit(train_set, validation_data=val_set, epochs=1)

#%% an encoder-decoder network for neural machine translation
from pathlib import Path

url = "https://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"
path = tfk.utils.get_file("spa-eng.zip", origin=url, cache_dir=".", extract=True)
text = (Path(path).with_name("spa-eng") / "spa.txt").read_text()

text = text.replace("¡", "").replace("¿", "")
pairs = [line.split("\t") for line in text.splitlines()]
np.random.shuffle(pairs)
sentences_en, sentences_es = zip(*pairs)

for i in range(3):
    print(sentences_en[i], "=>", sentences_es[i])

vocab_size = 1000
max_length = 50
text_vec_layer_en = tfk.layers.TextVectorization(vocab_size,
                                                 output_sequence_length=max_length)
text_vec_layer_es = tfk.layers.TextVectorization(vocab_size,
                                                 output_sequence_length=max_length)
text_vec_layer_en.adapt(sentences_en)
text_vec_layer_es.adapt([f"startofseq {s} endofseq" for s in sentences_es])

print(text_vec_layer_en.get_vocabulary()[:10])
print(text_vec_layer_es.get_vocabulary()[:10])

X_train = tf.constant(sentences_en[:100000])
X_val = tf.constant(sentences_en[100000:])
X_train_dec = tf.constant([f"startofseq {s}" for s in sentences_es[:100_000]])
X_val_dec = tf.constant([f"startofseq {s}" for s in sentences_es[100_000:]])
y_train = text_vec_layer_es([f"{s} endofseq" for s in sentences_es[:100_000]])
y_val = text_vec_layer_es([f"{s} endofseq" for s in sentences_es[100_000:]])

encoder_inputs = tfk.layers.Input(shape=[], dtype=tf.string)
decoder_inputs = tfk.layers.Input(shape=[], dtype=tf.string)

embedding_size = 128
encoder_input_ids = text_vec_layer_en(encoder_inputs)
decoder_input_ids = text_vec_layer_es(decoder_inputs)
encoder_embedding_layer = tfk.layers.Embedding(vocab_size, embedding_size, mask_zero=True)
decoder_embedding_layer = tfk.layers.Embedding(vocab_size, embedding_size, mask_zero=True)
encoder_embeddings = encoder_embedding_layer(encoder_input_ids)
decoder_embeddings = decoder_embedding_layer(decoder_input_ids)

encoder = tfk.layers.LSTM(512, return_state=True)
encoder_outputs, *encoder_state = encoder(encoder_embeddings)

decoder = tfk.layers.LSTM(512, return_sequences=True)
decoder_outputs = decoder(decoder_embeddings, initial_state=encoder_state)

output_layer = tfk.layers.Dense(vocab_size, activation="softmax")
y_proba = output_layer(decoder_outputs)

model = tfk.Model(inputs=[encoder_inputs, decoder_inputs], outputs=[y_proba])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
model.fit((X_train, X_train_dec), y_train, epochs=10,
          validation_data=((X_val, X_val_dec), y_val))

def translate(sentence_en):
    translation = ""
    for word_idx in range(max_length):
        X = np.array([sentence_en])
        X_dec = np.array(["startofseq " + translation])
        y_proba = model.predict((X, X_dec))[0, word_idx]
        pred_word_id = np.argmax(y_proba)
        pred_word = text_vec_layer_es.get_vocabulary()[pred_word_id]
        if pred_word == "endofseq":
            break
        translation += " " + pred_word
    return translation.strip()

print(translate("I like soccer"))
print(translate("I like soccer and going to the beach"))

# bidirectional rnns
encoder = tfk.layers.Bidirectional(tfk.layers.LSTM(256, return_sequences=True))
encoder_outputs, *encoder_state = encoder(encoder_embeddings)
encoder_state = [tf.concat(encoder_state[::2], axis=-1),
                 tf.concat(encoder_state[1::2], axis=-1)]

#%% attention mechanisms

encoder = tfk.layers.Bidirectional(tfk.layers.LSTM(256, return_sequences=True,
                                                   return_state=True))

attention_layer = tfk.layers.Attention()
attention_outputs = attention_layer([decoder_outputs, encoder_outputs])
output_layer = tfk.layers.Dense(vocab_size, activation="softmax")
y_proba = output_layer(attention_outputs)

model = tfk.Model(inputs=[encoder_inputs, decoder_inputs], outputs=[y_proba])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
model.fit((X_train, X_train_dec), y_train, epochs=10,
          validation_data=((X_val, X_val_dec), y_val))

print(translate("I like soccer and going to the beach"))


max_length = 50
embedding_size = 128
pos_embedding_layer = tfk.layers.Embedding(max_length, embedding_size)
batch_max_length_enc = tf.shape(encoder_embeddings)[1]
encoder_in = encoder_embeddings + pos_embedding_layer(tf.range(batch_max_length_enc))
batch_max_length_dec = tf.shape(decoder_embeddings)[1]
decoder_in = decoder_embeddings + pos_embedding_layer(tf.range(batch_max_length_dec))

class PositionalEncoding(tfk.layers.Layer):
    def __init__(self, max_length, embed_size, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        assert embedding_size % 2 == 0, "embedding size must be even"
        p, i = np.meshgrid(np.arange(max_length), 2 * np.arange(embedding_size // 2))
        pos_embedding = np.empty((1, max_length, embedding_size))
        pos_embedding[0, :, ::2] = np.sin(p / 10000 ** (i / embedding_size)).T
        pos_embedding[0, :, 1::2] = np.cos(p / 10000 ** (i / embedding_size)).T
        self.pos_encodings = tf.constant(pos_embedding.astype(self.dtype))
        self.supports_masking = True
    
    def call(self, inputs):
        batch_max_length = tf.shape(inputs)[1]
        return inputs + self.pos_encodings[:, :batch_max_length]
    
pos_embedding_layer = PositionalEncoding(max_length, embedding_size)
encoder_in = pos_embedding_layer(encoder_embeddings)
decoder_in = pos_embedding_layer(decoder_embeddings)


n = 2
num_heads = 8
dropout_rate = 0.1
num_units = 128
encoder_pad_mask = tf.math.not_equal(encoder_input_ids, 0)[:, tf.newaxis]
Z = encoder_in
for _ in range(n):
    skip = Z
    attention_layer = tfk.layers.MultiHeadAttention(num_heads=num_heads,
                                                    key_dim=embedding_size,
                                                    dropout=dropout_rate)
    Z = attention_layer(Z, value=Z, attention_mask=encoder_pad_mask)
    Z = tfk.layers.LayerNormalization()(tfk.layers.Add()([Z, skip]))
    skip = Z
    Z = tfk.layers.Dense(num_units, activation="relu")(Z)
    Z = tfk.layers.Dense(embedding_size)(Z)
    Z = tfk.layers.Dropout(dropout_rate)(Z)
    Z = tfk.layers.LayerNormalization()(tfk.layers.Add()([Z, skip]))

decoder_pad_mask = tf.math.not_equal(decoder_input_ids, 0)[:, tf.newaxis]
causal_mask = tf.linalg.band_part(
    tf.ones((batch_max_length_dec, batch_max_length_dec), tf.bool), -1, 0)

y_proba = tfk.layers.Dense(vocab_size, activation="softmax")(Z)
model = tfk.Model(inputs=[encoder_inputs, decoder_inputs], outputs=[y_proba])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
model.fit((X_train, X_train_dec), y_train, epochs=10,
          validation_data=((X_val, y_val), y_val))

#%% hugging face's transformers library
from transformers import AutoTokenizer, pipeline, TFAutoModelForSequenceClassification

classifier = pipeline("sentiment-analysis")
result = classifier("The actors were very convincing.")
print(result)
print(classifier(["I am from India." "I am from Iraq."]))

model_name = "huggingface/distilbert-base-uncased-finetuned-mnli"
classifier_mnli = pipeline("text-classification", model=model_name)
print(classifier_mnli("She loves me. [SEP] She loves me not."))

tokeniser = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

token_ids = tokeniser(["I like soccer. [SEP] We all love soccer!",
                       "Joe lived for a very long time. [SEP] Joe is old."],
                      padding=True, return_tensors="tf")
print(token_ids)

outputs = model(token_ids)
print(outputs)

y_probas = tfk.activations.softmax(outputs.logits)
print(y_probas)
y_pred = tf.argmax(y_probas, axis=1)
print(y_pred)

sentences = [("Sky is blue", "Sky is red"), ("I love her", "She loves me")]
X_train = tokeniser(sentences, padding=True, return_tensors="tf").data
y_train = tf.constant([0, 2])
loss = tfk.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(loss=loss, optimizer="nadam", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=2)




#%% Coding Exercises: Exercise 8

#%% Coding Exercises: Exercise 9

#%% Coding Exercises: Exercise 10

#%% Coding Exercises: Exercise 11