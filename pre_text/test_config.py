# 导包
import os
# 根目录文件路径
# __file__ 指的是动态的获取当前文件的路径
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


# 定义类
class Config:
    def __init__(self):
        self.root_path = ROOT_DIR if ROOT_DIR.endswith(os.sep) else ROOT_DIR + os.sep
        self.train_data_path = self.root_path + 'data/train.txt'
        self.test_data_path = self.root_path + 'data/test.txt'
        self.dev_data_path = self.root_path + 'data/dev.txt'


if __name__ == '__main__':
    config = Config()
    print(config)
    print(config.train_data_path)
    print(config.test_data_path)
    print(config.dev_data_path)
