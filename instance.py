import pandas as pd


class TemperatureInstance:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, header=0)
        self.y_mat = self.data.ix[:, 2:3].as_matrix().reshape(-1, 1)
        self.y_label = self.data.columns[2:3]
        # Skip timestamp data for now
        self.x_mat = self.data.ix[:, -2:].as_matrix()
        self.x_label = self.data.columns[-2:]
        self.num_feature = len(self.x_label)
