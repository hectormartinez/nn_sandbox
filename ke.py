from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data()
model = Sequential()
model.add(Dense(1, input_dim=len(X_train[0]), activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# generate dummy data
# train the model, iterating on the data in batches
# of 32 samples
model.fit(X_train, y_train, nb_epoch=10, batch_size=32)
model.evaluate(X_test,y_test)