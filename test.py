from tensorflow import keras
from vpnn.models import vpnn
import tensorflow as tf
import numpy as np
from models.resnet import resnet_50

(X_train_full, y_train_full), (X_test,
                               y_test) = keras.datasets.fashion_mnist.load_data()
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True) + 1e-7
X_train = (X_train - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]

model = resnet_50()
model.build(input_shape=(None, 28, 28, 1))
model.summary()
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)
model.fit(
    x=X_train,
    y=y_train,
    epochs=100,
    validation_data=(X_test, y_test)
)
print(model)
