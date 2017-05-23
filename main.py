from instance import TemperatureInstance
from predictor import TemperaturePredictor
import keras
import numpy as np
import sys
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
import matplotlib.pylab as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_PATH = r'C:\Users\coffee\Desktop\8\cs570\tempredict\data\processed_data_inner.csv'


def main():
    print("python:{}, keras:{}, tensorflow: {}".format(sys.version, keras.__version__, tf.__version__))
    instance = TemperatureInstance(DATA_PATH)
    # predictor = TemperaturePredictor(instance)
    # predictor.predict()

    xntrn = round(len(instance.x_mat) * 0.9)

    x_train = instance.x_mat[0:xntrn]
    x_test = instance.x_mat[xntrn:]
    y_train = instance.y_mat[0:xntrn]
    y_test = instance.y_mat[xntrn:]

    in_neurons = len(instance.x_label)
    # in_neurons = 2
    out_neurons = len(instance.y_label)
    # out_neurons = 2
    hidden_neurons = 100

    model = Sequential()
    model.add(LSTM(hidden_neurons, return_sequences=False,
                   input_shape=(None, in_neurons)))
    model.add(Dense(out_neurons, input_dim=hidden_neurons))
    model.add(Activation("linear"))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")

    model.summary()
    # print(np.shape(x_train))
    # print(np.shape(y_train))
    xx_train = np.array([x_train[i:i + 10] for i in range(len(x_train) - 10)])
    yy_train = np.array([y for y in y_train[10:]])
    # print(np.shape(xx_train))
    # print(np.shape(yy_train))
    xx_test = np.array([x_test[i:i + 10] for i in range(len(x_test) - 10)])
    yy_test = np.array([y for y in y_test[10:]])
    model.fit(xx_train, yy_train, batch_size=5, epochs=100, validation_split=0.05)

    predicted = model.predict(xx_test)
    rmse = np.sqrt(((predicted - yy_test) ** 2).mean(axis=0))
    print(rmse)

    plt.rcParams["figure.figsize"] = (17, 9)
    plt.plot(predicted[:100][:, 0], "--")
    plt.plot(predicted[:100][:, 1], "--")
    plt.plot(yy_test[:100][:, 0], ":")
    plt.plot(yy_test[:100][:, 1], ":")
    plt.legend(["Prediction 0", "Prediction 1", "Test 0", "Test 1"])
    plt.show()

if __name__ == '__main__':
    main()
