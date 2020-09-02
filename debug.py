# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

import os
from pathlib import Path
import string
import random

CORPUS_FOLDER = r'F:\Documenti\DL_Bergamo\testi_italiano'

class Corpus:
    def __init__(self, folder):
        self.folder = folder
        self.good_chars = string.ascii_lowercase + string.digits + ',.:; \n"'
        self.load()
        self.char_to_num_dict, self.num_to_char_dict = self.build_dictionaries()
        
    def load(self):
        big_string = ''
        folder = Path(self.folder)
        for file in os.listdir(self.folder):
            with open(folder / file, 'r', encoding='utf-8') as f:
                big_string += f.read()
        big_string = big_string.lower()
        self.big_string = ''.join([char for char in big_string if char in self.good_chars])
        self.n = len(self.big_string)
        
    def build_dictionaries(self):
        c_2_n, n_2_c = {}, {}
        charset = set(self.big_string)
        for i, char in enumerate(charset):
            c_2_n[char] = i+1
            n_2_c[i+1] = char
        return c_2_n, n_2_c
        
    def get_dictionary_cardinality(self):
        return len(self.char_to_num_dict)
        
    def chars_to_nums(self, chars):
        nums = [self.char_to_num_dict[char] for char in chars]
        return nums
    
    def nums_to_chars(self, nums):
        chars = [self.num_to_char_dict[num] for num in nums]
        return ''.join(chars)
        
    def take_sample(self, length):
        idx = random.randint(0, self.n-length)
        chars = self.big_string[idx:idx+length+1]
        nums = self.chars_to_nums(chars)
        x, y =  nums[:-1], nums[-1]
        return x, y
        
    def take_batch(self, size, max_len):
        xs, ys = [], []
        lengths = random.choices(range(1, max_len+1), k=size)
        for length in lengths:
            x, y = self.take_sample(length)
            xs.append(x)
            ys.append(y-1)
        
        return pad_sequences(xs), tf.one_hot(ys, self.get_dictionary_cardinality())
        
        

# %%

    

corpus = Corpus(r'F:\Documenti\DL_Bergamo\testi_italiano')

def generator(corpus=corpus, num_batches=100, batch_size=32, max_seq_len=100):
    while True:
        yield corpus.take_batch(batch_size, max_seq_len)

# %%
N = corpus.get_dictionary_cardinality()
model = keras.Sequential(
    [
        keras.Input(batch_size=32, shape=(None,)),
        keras.layers.Embedding(input_dim=N+1, output_dim=4, mask_zero=True),
        keras.layers.GRU(512, return_sequences=True),
        keras.layers.GRU(512, return_sequences=True),
        keras.layers.GRU(512),
        keras.layers.Dense(N, activation='softmax')
    ]
)


train_ds = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.float32))

val_ds = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.float32))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2), loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)
# %%
model.fit(x=train_ds, validation_data=val_ds.take(50), epochs=10, steps_per_epoch=100)
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
