#%%
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Identity

(x_tr, y_tr), (x_val, y_val) = keras.datasets.fashion_mnist.load_data()
# %%
x_tr = x_tr / 255
x_val = x_val / 255

x_tr = x_tr.reshape(-1, 28*28)
x_val = x_val.reshape(-1, 28*28)

y_tr = tf.one_hot(y_tr, 10)
y_val = tf.one_hot(y_val, 10)
# %%
base_model = keras.Sequential([
    Dense(100, activation='relu', name='1'),
])

top_model = keras.Sequential([
    Dense(10, activation='softmax')
])

model = keras.Sequential([
    base_model,
    top_model
])
# %%

model.compile(optimizer=Adam(1e-2), loss='categorical_crossentropy', metrics=['accuracy'])

#%%
model.fit(x_tr, y_tr, batch_size=32, epochs=10, validation_split=0.2)
# %%
layer_names = [layer.name for layer in base_model.layers]
# %%
base_model.add(Dense(100, activation='relu', kernel_initializer=Identity(), name='4'))

for layer_name in layer_names:
    layer = base_model.get_layer(name=layer_name)
    layer.trainable = False
# %%
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_tr, y_tr, batch_size=32, epochs=10, validation_split=0.2)

# %%
