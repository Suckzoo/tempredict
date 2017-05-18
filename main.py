from .instance import TemperatureInstance
from .predictor import TemperaturePredictor

DATA_PATH = 'data/processed_data_inner.csv'


def main():
    instance = TemperatureInstance(DATA_PATH)
    predictor = TemperaturePredictor(instance)

if __name__ == '__main__':
    main()
