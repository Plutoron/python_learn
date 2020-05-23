import tensorflow as tf

import load_fashion_datasets

# fashion = tf.keras.datasets.fashion_mnist
# (x_train, y_train),(x_test, y_test) = fashion.load_data()
(x_train, y_train), (x_test, y_test) = load_fashion_datasets.load_localData('./datasets/fashion_mnist')

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

model.compile(optimizer = 'adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
              metrics = "sparse_categorical_accuracy") 

model.fit(x_train, y_train, batch_size = 32, epochs = 4, validation_split = 0.2, validation_freq = 1)

model.summary()