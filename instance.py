import pandas as pd


class TemperatureInstance:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, header=0)
        self.y_mat = self.data.ix[:, 1:3].as_matrix().reshape(-1, 2)
        self.y_label = self.data.columns[1:3]
        # Skip timestamp data for now
        self.x_mat = self.data.ix[:, 3:-1].as_matrix()
        self.x_label = self.data.columns[3:-1]
        self.num_feature = len(self.x_label)
