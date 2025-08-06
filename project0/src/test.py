import os

from src.predict import model_predict
from src.train import PowerLoadModel, analysis_data, data_processing, model_train

if __name__ == '__main__':
    # 1.加载数据集
    input_file = os.path.join('../data', 'train.csv') # 拼接文件
    model = PowerLoadModel(input_file)
    # 2.分析数据
    analysis_data(model.data_source)
    # 3.特征工程
    x_train,y_train = data_processing(model.data_source)
    # processed_data, feature_cols = feature_engineering(model.data_source, model.logfile)
    # 4.模型训练、模型评价与模型保存
    model_train(x_train,y_train)
    # model_train(processed_data, feature_cols, model.logfile)
    model_predict('test.csv')