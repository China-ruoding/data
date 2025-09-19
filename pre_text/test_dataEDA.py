# 导包
import pandas as pd
from test_config import Config
from collections import Counter


# 构建函数
# 读取数据
# 获取所需要的列
# 统计信息
config = Config()

def data_eda(path):
    df_data = pd.read_csv(
        path,
        sep='\t',
        header=None,
        names=['text','label']
    )
    print(f'样例-前十行 =\n {df_data.head(10)}')
    print(f'样本总数 = {len(df_data)}')
    print(f'标签列：{df_data['label'][:100]}')
    label_counter = Counter(df_data['label'])
    print(f'标签分布{label_counter}')
    # 标签分布Counter({3: 18000, 4: 18000, 1: 18000, 7: 18000, 5: 18000, 9: 18000, 8: 18000, 2: 18000, 6: 18000, 0: 18000})
    for key,value in label_counter.items():
        print(f'总量{len(df_data)},标签{key},出现次数{value},占比{value/(len(df_data)):.2%}')
    df_data['text_length'] = df_data['text'].apply(lambda x: len(x))
    print(f'新增text_length前十行=\n {df_data.head(10)}')
    print(f'text_length的统计信息：{df_data["text_length"].describe()}')
    print(f'平均长度：{df_data["text_length"].mean()}')
    print(f'最大标准差：{df_data["text_length"].std()}')
    print(f'最大长度：{df_data['text_length'].max()}')
    print(f'最小长度：{df_data["text_length"].min()}')
    




if __name__ == '__main__':
    data_eda(config.train_data_path)