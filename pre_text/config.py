# 导包
import os

Root_dir = os.path.dirname(os.path.dirname(__file__))


class Config:
    def __init__(self):
        self.root_path = Root_dir if Root_dir.endswith(os.sep) else Root_dir + os.sep
        self.train_data_path = self.root_path + 'data/train.txt'
        self.test_data_path = self.root_path + 'data/test.txt'
        self.dev_data_path = self.root_path + 'data/dev.txt'


if __name__ == '__main__':
    config = Config()
    print(config)
    print(config.train_data_path)
    print(config.test_data_path)
    print(config.dev_data_path)