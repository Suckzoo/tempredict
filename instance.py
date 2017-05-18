import pandas as pd


class TemperatureInstance:
    def __init__(self, file_path):
        with pd.read_csv(file_path, header=0) as data:
            self.y_mat = data.ix[:, 1:3].as_matrix().reshape(-1, 2)
            self.y_label = data.columns[1:3]
            # Skip timestamp data for now
            self.x_mat = data.ix[:, 3:-1].as_matrix()
            self.x_label = data.columns[3:-1]
