# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from random import shuffle
from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.utils import Sequence

import os
from pathlib import Path
import string
import random

d = tfds.load(name='tiny_shakespeare')['train']
val = tfds.load(name='tiny_shakespeare')['validation']
big_string_train = next(iter(d))['text'].numpy().decode("utf-8")
big_string_val = next(iter(val))['text'].numpy().decode("utf-8")
chars = set(big_string_train)
dict_size = len(chars)

def get_temporal_series(l, n):
    xs = [l[i:i+n] for i in range(len(l)-n)]
    ys = [l[i+1:i+n+1] for i in range(len(l)-n)]
    return np.asarray(xs), np.asarray(ys)
    
char_to_num = {char: num for (num, char) in enumerate(chars)}
num_to_char = {num: char for (num, char) in enumerate(chars)}

big_string_train = [char_to_num[char] for char in big_string_train]
big_string_val = [char_to_num[char] for char in big_string_val]

# x_train, y_train =get_temporal_series(big_string_train, 100)
# x_val, y_val =get_temporal_series(big_string_val, 100)

class Dataset(Sequence):
    def __init__(self, x, dict_size, batch_size=32):
        super().__init__()
        self.x = x
        self.dict_size = dict_size
        self.reset()
        
    def reset(self):
        n = len(self.x)
        max_init_idx = n - 100*100 - 100
        self.current_idx = random.randint(0, max_init_idx)
        
    def __len__(self):
        return 100
        
    def __getitem__(self, idx):
        xs, ys = [], []
        idx = self.current_idx
        step = 3
        for idx in (self.current_idx + idx*step for idx in range(0, 32)):
            xs.append(self.x[idx:idx+100])
            ys.append(self.x[idx+1:idx+100+1])
        
        self.current_idx += 100
        x = np.stack(xs)
        y = np.stack(ys)
        return x, tf.one_hot(y, self.dict_size, axis=-1)
    
    def on_epoch_end(self):
        self.reset()

# %%
N = len(chars)
model = keras.Sequential(
    [
        keras.Input(batch_size=32, shape=(None,)),
        keras.layers.Embedding(input_dim=N, output_dim=4, mask_zero=False),
        keras.layers.GRU(512, return_sequences=True, stateful=True),
        keras.layers.GRU(512, return_sequences=True, stateful=True),
        keras.layers.GRU(512, return_sequences=True, stateful=True),
        keras.layers.Dense(N, activation='softmax')
    ]
)

callbacks = [
    EarlyStopping(patience=5),
    CSVLogger('logger.csv'),
    ModelCheckpoint('checkpoint', save_model=True, restore_model=True)
]

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2), loss='categorical_crossentropy', metrics=['accuracy'])

train_ds = Dataset(big_string_train, dict_size)
val_ds = Dataset(big_string_val, dict_size)
# %%
model.fit(x=train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks, shuffle=False)
model.save('checkpoint.tf')
# %%
model = keras.models.load_model('checkpoint.tf')
# %%
class Generator:
    def __init__(self, model, chars_to_nums, nums_to_chars, initial_context=' '):
        self.model = model
        self.context = chars_to_nums(initial_context)
        self.chars_to_nums = chars_to_nums
        self.nums_to_chars = nums_to_chars
        
    def choose_char_from_probs(self, probs):
        return random.choices(range(1, len(probs)+1), weights=probs, k=1)[0]
        
        
    def next(self):
        next_char_probs = self.model(np.array(self.context[-100:]).reshape(1, -1)).numpy().ravel()
        next_char = self.choose_char_from_probs(next_char_probs)
        self.context.append(next_char)
        return self.nums_to_chars([next_char])
        
generator = Generator(model, corpus.chars_to_nums, corpus.nums_to_chars, 'ieri sono')
for _ in range(50):
    print(generator.next())
# %%
