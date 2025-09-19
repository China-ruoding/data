import pandas as pd
from rf_config import Config
import jieba

config = Config()

def process_data(datapath,processed_datapath):
    df_data = pd.read_csv(
        datapath,
        sep='\t',
        header=None,
        names = ['text', 'label']
    )

    print(f'原始数据集前五行 =\n {df_data.head()}')
    df_data['words'] = df_data['text'].apply(lambda x: " ".join(jieba.lcut(x)[:30]))
    print(f'样例-前5行 = \n {df_data.head()}')
    df_data.to_csv(
        processed_datapath,
        sep='\t',
        index=False,
        header=True
    )
    print(f'已保存到 = {processed_datapath}')




if __name__ == '__main__':
    process_data(config.train_dir,config.process_train_data_path)
    process_data(config.test_dir,config.process_test_data_path)
    process_data(config.dev_dir,config.process_dev_data_path)