import os

import joblib
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

from src.train import PowerLoadModel, data_processing
from  util.commonUtil import model_view_AUC,model_hunxiao


def model_predict(path):
    input_file = os.path.join('../data', path)
    model = PowerLoadModel(input_file)
    x_test, y_test = data_processing(model.data_source)
    model = joblib.load('../model/xgb.pkl')
    y_pre = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]

    # 评价指标
    accuracy = accuracy_score(y_test, y_pre)
    precision = precision_score(y_test, y_pre)
    recall = recall_score(y_test, y_pre)
    f1 = f1_score(y_test, y_pre)
    auc = roc_auc_score(y_test,y_pred_proba)

    print(f'准确率: {accuracy}')
    print(f'精确率: {precision}')
    print(f'召回率: {recall}')
    print(f'F1分数: {f1}')
    print(f'AUC:{auc}')

    model_view_AUC(x_test,y_test,model)
    model_hunxiao(y_test,y_pre)


