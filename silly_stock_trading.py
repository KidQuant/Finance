import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import requests
import numpy as np

x_train = np.random.random((1000,20))
y_train = keras.utils.to_categorical(np.random.randint(10,size=(1000,)), num_classes=10)
x_test = np.random.random((100,20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100,1)), num_classes=10)

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=120)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import requests
url = 'http://nypost.com/horoscope/aries-12-01-2013/'
page = requests.get(url)
page.ok

from urllib.request import urlopen
from bs4 import BeautifulSoup
url = 'http:/nypost.com/horoscope/taurus-12-01-2017/'
content = urlopen(url)
soup = BeautifulSoup(content)
