import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from datetime import datetime

logdir = "C:\\Users\\Mayur\\Documents\\Code\\Projects\\Heart_Disease\\logs\\" + datetime.now(
).strftime("%Y%m%d-%H%M%S")

data = pd.read_csv(
    'C:\\Users\\Mayur\\Documents\\Code\\Projects\\Heart_Disease\\Data\\heart.csv'
)

data['thal'] = pd.Categorical(data['thal'])
data['thal'] = data.thal.cat.codes

target = data.pop('target')

train_data, test_data, train_target, test_target = train_test_split(
    data, target, test_size=0.33)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_data.values, test_target.values))

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_data.values, train_target.values))

train_dataset = train_dataset.batch(1)
test_dataset = test_dataset.batch(1)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='relu'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_dataset,
          epochs=15,
          verbose=0,
          callbacks=[tensorboard_callback],
          validation_data=test_dataset)
