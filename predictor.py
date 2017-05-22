from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM


class TemperaturePredictor:
    in_out_neurons = 2
    hidden_neurons = 300

    def __init__(self, instance):
        self.instance = instance
        model = Sequential()
        model.add(LSTM(self.hidden_neurons, return_sequences=False,
                            input_shape=(None, self.in_out_neurons)))
        model.add(Dense(in_out_neurons, input_dim=hidden_neurons))
        model.add(Activation("linear"))
        model.compile(loss="mean_squared_error", optimizer="rmsprop")
        self.model = model

    def predict(self):
        pass
