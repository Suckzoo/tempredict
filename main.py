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
    files = os.listdir(DATA_DIR)
    files.remove('README.txt')

    # predictor = TemperaturePredictor(instance)
    # predictor.predict()
    intervals = [1, 12, 36]
    idx = 0
    predict_interval = intervals[idx]  # 1, 12, 36
    x_raw = []
    y_raw = []
    x_label = None
    y_label = None
    model_name = None
    model_loaded = False

    for f in files:
        instance = TemperatureInstance(os.path.join(DATA_DIR, f))
        x_raw.append(instance.x_mat[:])
        y_raw.append(instance.y_mat[:])
        if x_label is None:
            model_name = "model_t" + str(idx) + "_" + instance.model_label + ".h5"
            x_label = instance.x_label
            y_label = instance.y_label

    in_neurons = len(x_label)
    out_neurons = len(y_label)
    data_interval = 12
    hidden_neurons = data_interval ** 2

    try:
        model = load_model(model_name)
        model_loaded = True
    except Exception:
        model = Sequential()
        model.add(LSTM(hidden_neurons, return_sequences=False, input_shape=(None, in_neurons)))
        model.add(Dense(data_interval))
        model.add(Dense(out_neurons))
        model.compile(loss="mean_squared_error",
                      optimizer=keras.optimizers.RMSprop(lr=0.00005, rho=0.9, epsilon=1e-08, decay=0.0))
        model.summary()

    x_data = []
    y_data = []
    # test_index = random.randint(0, len(x_raw) - 1)
    # test_index = len(x_raw) - 1
    # x_t_data = x_raw.pop(test_index)
    # y_t_data = y_raw.pop(test_index)
    tests = [0, 2, 6, 10, 13, 17, 21, 25, 28, 32, 36]
    indices = [i for i in range(0, 40) if i not in tests]
    x_t_datas = [x_raw[i] for i in tests]
    y_t_datas = [y_raw[i] for i in tests]
    test_index = 10
    x_t_data = x_t_datas[test_index]
    y_t_data = y_t_datas[test_index]
    # x_raw = x_raw[:]
    # y_raw = y_raw[:]
    x_raw = [x_raw[i] for i in indices]
    y_raw = [y_raw[i] for i in indices]
    '''
    index of categories
    0-2: YouTube2K
    2-6: YouTube4K
    6-10: Net300Mbps
    10-13: Net100Mbps
    13-17: Net10Mbps
    17-21: Net1Mbps
    21-25: yogoi
    25-28: HIT
    28-32: SkypeBackCam
    32-36: VideoRec720p
    36-40: VideoRec1080p
    '''
    for idx in range(len(x_raw)):
        x = x_raw[idx]
        y = y_raw[idx]
        x_data += [x[i:i + data_interval] for i in range(len(x) - data_interval - predict_interval)]
        y_data += [y_ for y_ in y[data_interval + predict_interval:]]
    x_t_data = [x_t_data[i:i + data_interval] for i in range(len(x_t_data) - data_interval - predict_interval)]
    y_t_data = [y for y in y_t_data[data_interval + predict_interval:]]

    xx_train = np.array(x_data)
    yy_train = np.array(y_data)
    xx_test = np.array(x_t_data)
    yy_test = np.array(y_t_data)
    print("training x: ", np.shape(xx_train))
    print("training y: ", np.shape(yy_train))
    print("testing x: ", np.shape(xx_test))
    print("testing y: ", np.shape(yy_test))
    if not model_loaded:
        model.fit(xx_train, yy_train, batch_size=len(x_raw), epochs=40, verbose=1)
        model.save(model_name)
    # model.save_weights("model_w.h5")

    predicted = model.predict(xx_test)
    rmse = np.sqrt(((predicted - yy_test) ** 2).mean(axis=0))
    print(rmse)

    f = plt.figure()
    ax = f.add_subplot(111)
    plt.plot(predicted[:][:, 0], "--")
    # plt.plot(predicted[:100][:, 1], "--")
    plt.plot(yy_test[:][:, 0], ":")
    # plt.plot(yy_test[:100][:, 1], ":")
    plt.text(0.05, 0.05, "rms: " + str(rmse[0]), va="bottom", ha="left", transform=ax.transAxes)
    plt.rcParams["figure.figsize"] = (17, 9)
    legend = ["Pred_" + l for l in y_label] + ["Real_" + l for l in y_label]
    plt.legend(legend)
    plt.show()


if __name__ == '__main__':
    main()
