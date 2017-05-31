import pandas as pd


class TemperatureInstance:
    def __init__(self, file_path):
        # 0 Time
        # 1 AverageTemp
        # 2 MaxTemp
        # 3 User%       O
        # 4 System%     O
        # 5 IOW%
        # 6 IRQ%
        # 7 User
        # 8 Nice
        # 9 Sys
        # 10 Idle
        # 11 IOW
        # 12 IRQ
        # 13 SIRQ
        # 14 Sum7
        # 15 CPU%       O
        # 16 #THR
        # 17 VSS
        # 18 RSS
        # 19 CPUfreq    O
        # 20 netTx      O
        # 21 netRx      O
        # 22 batTemp    O
        # 23 cpuTemp    O
        x_indices = [3, 4, 15, 19, 22, 23]
        self.model_label = "_".join([str(x_i) for x_i in x_indices])

        self.data = pd.read_csv(file_path, header=0)

        # Skip timestamp data for now
        self.x_mat = self.data.ix[:, x_indices].as_matrix()
        self.x_label = self.data.columns[x_indices]

        self.y_mat = self.data.ix[:, 2:3].as_matrix()  # .reshape(-1, 1)
        self.y_label = self.data.columns[2:3]
