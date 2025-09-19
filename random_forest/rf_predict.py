import pandas as pd
from rf_config import Config
import pickle
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

config = Config()

with open(config.rf_model_path,'rb') as f:
    rf = pickle.load(f)

with open(config.tfidf_model_save_path,'rb') as f:
    tfidf = pickle.load(f)

df_data = pd.read_csv(config.process_dev_data_path,sep='\t')
words = df_data['words']
features = tfidf.transform(words)

y_pred = rf.predict(features)

print(f'准确率：{accuracy_score(df_data['label'],y_pred)}')
print(f'精确率：{precision_score(df_data['label'],y_pred,average='macro')}')
print(f'召回率：{recall_score(df_data['label'],y_pred,average='macro')}')
print(f'F1值：{f1_score(df_data['label'],y_pred,average='macro')}')

df_data['pred_label'] = y_pred
df_data.to_csv(config.model_tex_save_path,sep='\t',index=False)