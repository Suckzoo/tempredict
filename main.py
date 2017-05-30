from instance import TemperatureInstance
from predictor import TemperaturePredictor
import random
import keras
import numpy as np
import sys
import os
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
import matplotlib.pylab as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_DIR = os.path.join(os.curdir, "data")


def main():
    print("python:{}, keras:{}, tensorflow: {}".format(sys.version, keras.__version__, tf.__version__))
    instance = TemperatureInstance(os.path.join(DATA_DIR, "data_01.csv"))
    files = os.listdir(DATA_DIR)
    files.remove('processed_data_inner.csv')
    files.remove('processed_data_outer.csv')
    files.remove('data_01.csv')
    files.remove('README.txt')

    # predictor = TemperaturePredictor(instance)
    # predictor.predict()

    x_raw = [instance.x_mat[:]]
    y_raw = [instance.y_mat[:]]
    x_label = instance.x_label
    y_label = instance.y_label

    for f in files:
        instance = TemperatureInstance(os.path.join(DATA_DIR, f))
        x_raw.append(instance.x_mat[:])
        y_raw.append(instance.y_mat[:])

    in_neurons = len(x_label)
    out_neurons = len(y_label)
    data_interval = 12
    hidden_neurons = data_interval * 10

    model = Sequential()
    model.add(LSTM(hidden_neurons, return_sequences=False, input_shape=(None, in_neurons)))
    model.add(Dense(hidden_neurons))
    model.add(Dense(out_neurons))
    model.compile(loss="mean_squared_error",
                  optimizer=keras.optimizers.RMSprop(lr=0.00005, rho=0.9, epsilon=1e-08, decay=0.0))
    model.summary()

    x_data = []
    y_data = []
    test_index = random.randint(0, len(x_raw)-1)
    test_index = len(x_raw)-1
    # x_t_data = x_raw.pop(test_index)
    # y_t_data = y_raw.pop(test_index)
    x_t_data = x_raw[10]
    y_t_data = y_raw[10]
    x_raw = x_raw[10:11]
    y_raw = y_raw[10:11]
    for idx in range(len(x_raw)):
        x = x_raw[idx]
        y = y_raw[idx]
        x_data += [x[i:i + data_interval] for i in range(len(x) - data_interval)]
        y_data += [y_ for y_ in y[data_interval:]]
    x_t_data = [x_t_data[i:i + data_interval] for i in range(len(x_t_data) - data_interval)]
    y_t_data = [y for y in y_t_data[data_interval:]]

    xx_train = np.array(x_data)
    yy_train = np.array(y_data)
    xx_test = np.array(x_t_data)
    yy_test = np.array(y_t_data)
    print("training x: ", np.shape(xx_train))
    print("training y: ", np.shape(yy_train))
    print("testing x: ", np.shape(xx_test))
    print("testing y: ", np.shape(yy_test))
    model.fit(xx_train, yy_train, batch_size=1, epochs=25, verbose=1)
    model.save("model.h5")
    model.save_weights("model_w.h5")

    predicted = model.predict(xx_test)

    plt.rcParams["figure.figsize"] = (17, 9)
    plt.plot(predicted[:][:, 0], "--")
    # plt.plot(predicted[:100][:, 1], "--")
    plt.plot(yy_test[:][:, 0], ":")
    # plt.plot(yy_test[:100][:, 1], ":")
    legend = ["Pred_" + l for l in y_label] + ["Real_" + l for l in y_label]
    plt.legend(legend)
    plt.show()

if __name__ == '__main__':
    main()
