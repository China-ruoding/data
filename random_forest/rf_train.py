import pandas as pd
from sklearn.model_selection import train_test_split

from rf_config import Config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from tqdm import tqdm
import pickle


# 初始化配置
config = Config()
print(f'配置 训练数据路径：{config.process_train_data_path}')
print(f'配置 停用词路径：{config.stop_words_path}')
print(f'配置 RF模型保存路径：{config.rf_model_path}')
print(f'配置 TFIDF模型保存路径：{config.tfidf_model_save_path}')

# 读取训练数据
df_data = pd.read_csv(config.process_train_data_path, sep='\t')[:20000]
print(f'数据 数据形状：{df_data.shape}')
print(f'数据 列：{list(df_data.columns)}')
print(f'数据 前5行样例：\n {df_data.head()}')
print(f'数据 缺失值统计：\n {df_data.isna().sum()}')



# 提取文本与标签列
words = df_data['words']
labels = df_data['label']
print(f'数据 文本数量：{len(words)},标签数量：{len(labels)}')
print(f'数据 标签值分布：\n {labels.value_counts()}\n')


# 读取停用词
stop_words = [line.strip() for line in open(config.stop_words_path,encoding='utf-8').readlines()]
print(f'停用词：数量：{len(stop_words)}')
print(f'停用词：前十个：{stop_words[:10]}')

# 构建TF-IDF向量器
tfidf = TfidfVectorizer(stop_words=stop_words)
print(f'[TF-IDF]参数 ：{tfidf}')

# 文本特征化
features = tfidf.fit_transform(words)
print(f'特征形状：{features.shape}(样本数，词汇表大小)')
print(f'特征稀疏度：非零元素/总元素 = {features.nnz}/{features.shape[0]*features.shape[1]}')

# 划分训练集/测试集
x_train,x_test,y_train,y_test = train_test_split(
    features,
    labels,
    test_size=0.2,
    random_state=28
)

print(f'划分 训练集形状：{x_train.shape},测试集形状：{x_test.shape}')
print(f'划分 训练集标签数：{len(y_train)},测试集标签数：{len(y_test)}')

# 初始化并训练随机森林
rf = RandomForestClassifier()
print(f'模型 随机森林参数:{rf}')
for i in tqdm(range(1),desc='随机森林训练进度....'):
    print(f'训练 第{i+1}轮开始')
    rf.fit(x_train,y_train)
    print(f'训练 第{i+1}轮结束')

# 推理预测
y_pred = rf.predict(x_test)
print(f'预测 样例前10个预测：{list(y_pred[:10])}')
print(f'预测 真实前10个标签：{list(y_test[:10])}')

# 评估指标
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred,average='macro')
rec = recall_score(y_test, y_pred,average='macro')
f1 = f1_score(y_test,y_pred,average='macro')
print(f'准确率：{acc}')
print(f'精确率：{pre}')
print(f'召回率：{rec}')
print(f'F1值：{f1}')
print(f'评估报告：\n {classification_report(y_test, y_pred)}')
print(f'混淆矩阵：\n {confusion_matrix(y_test, y_pred)}')

with open(config.rf_model_path,'wb') as f:
    pickle.dump(rf,f)

print(f'保存 随机森林模型已保存到：{config.rf_model_path}')

with open(config.tfidf_model_save_path,'wb') as f:
    pickle.dump(tfidf,f)
print(f'保存 TF-IDF向量器已保存：{config.tfidf_model_save_path}')



if __name__ == '__main__':
    pass