from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()
model = Sequential()
model.add(Dense(1, input_dim=784, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# generate dummy data
# train the model, iterating on the data in batches
# of 32 samples

X2 = []
for x in X_train:
    r  = []
    for xi in x:
        r.extend(xi)
    X2.append(r)

X2 = np.array(X2)

X3 = []
for x in X_test:
    r  = []
    for xi in x:
        r.extend(xi)
    X3.append(r)

X3 = np.array(X3)

model.fit(X2, y_train, nb_epoch=10, batch_size=32)
model.evaluate(X3,y_test)