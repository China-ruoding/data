# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns


def data_preprocessing(path):
    """
    1.获取数据
    2.返回数据
    :param path:
    :return:
    """
    data = pd.read_csv(path)
    return data


def model_view_AUC(x_test,y_test,model):
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False
    # 设置负号显示
    plt.plot(fpr, tpr, label=f'ROC 曲线 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率')
    plt.ylabel('真正例率')
    plt.title('受试者工作特征曲线')
    plt.legend(loc='lower right')
    plt.show()

def model_hunxiao(y_test,y_pred):

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.title('混淆矩阵')
    plt.show()

if __name__ == '__main__':

    data = data_preprocessing(r'../data/train.csv')
    print(data)
