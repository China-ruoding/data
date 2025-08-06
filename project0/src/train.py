import os
import datetime
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from util.logUtil import Logger
from util.commonUtil import data_preprocessing
import joblib
from sklearn.preprocessing import StandardScaler


# 分析数据
def analysis_data(data):
    columns_tab = data.columns
    n_list = []
    for c in columns_tab:
        if c == 'Attrition':
            continue
        if data[c].dtypes == 'int64':
            n_list.append(roc_auc_score(data['Attrition'],data[c]))
    return n_list


# 数据处理、特征工程
def data_processing(data, logger):
    logger.info("=========开始进行特征工程===================")
    # 创建映射表
    map_Bus = {
        'Non-Travel': 0,
        'Travel_Rarely': 1,
        'Travel_Frequently': 2,
        'Human Resources': 1,
        'Research & Development': 2,
        'Sales': 3,
        'Divorced': 0,
        'Single': 1,
        'Married': 2,
        'Male': 0,
        'Female': 1
    }

    # 映射替换为数值形式
    data['MaritalStatus'] = data['MaritalStatus'].map(map_Bus)
    data['BusinessTravel'] = data['BusinessTravel'].map(map_Bus)
    data['Department'] = data['Department'].map(map_Bus)
    data['Gender'] = data['Gender'].map(map_Bus)

    data_dum = pd.get_dummies(data).astype(int)

    x = data_dum.drop('Attrition',axis = 1)
    y = data_dum['Attrition']

    std = StandardScaler()
    x = std.fit_transform(x)

    return x,y


# 训练模型
def model_train(x_train,y_train):
    # 创建逻辑回归模型
    model_log = LogisticRegression(max_iter=1000)
    model_log.fit(x_train, y_train)
    joblib.dump(model_log, '../model/xgb.pkl')



class PowerLoadModel(object):
    def __init__(self, filename):
        # 配置日志记录
        logfile_name = "train_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.logfile = Logger('../', logfile_name).get_logger()
        # 获取数据源
        self.data_source = data_preprocessing(filename)


if __name__ == '__main__':
    # 1.加载数据集
    input_file = os.path.join('../data', 'train.csv') # 拼接文件
    model = PowerLoadModel(input_file)
    # 2.分析数据
    analysis_data(model.data_source)
    # 3.特征工程
    x_train,y_train = data_processing(model.data_source, model.logfile)
    # processed_data, feature_cols = feature_engineering(model.data_source, model.logfile)
    # 4.模型训练、模型评价与模型保存
    model_train(x_train,y_train)
    # model_train(processed_data, feature_cols, model.logfile)