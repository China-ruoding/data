# 导包
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))


class Config:
    def __init__(self):
        self.root_dir = ROOT_DIR if ROOT_DIR.endswith(os.sep) else ROOT_DIR + os.sep
        self.train_dir = self.root_dir + 'data/train.txt'
        self.test_dir = self.root_dir + 'data/test.txt'
        self.dev_dir = self.root_dir + 'data/dev.txt'

        self.class_doc_path = self.root_dir + 'data/class.txt'
        self.stop_words_path = self.root_dir + 'data/stopwords.txt'

        self.process_train_data_path = self.root_dir + 'random_forest/data/train_process.txt'
        self.process_test_data_path = self.root_dir + 'random_forest/data/test_process.txt'
        self.process_dev_data_path = self.root_dir + 'random_forest/data/dev_process.txt'

        self.rf_model_path = self.root_dir + 'random_forest/save_model/rf_model.pkl'
        self.tfidf_model_save_path = self.root_dir + 'random_forest/save_model/tfidf_model.pkl'

        self.model_tex_save_path =self.root_dir+'random_forest/result/predict_result.txt'



if __name__ == '__main__':
    config = Config()
    print(config.rf_model_path)
    print(config.tfidf_model_save_path)