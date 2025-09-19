import jieba
import pickle
from rf_config import Config

conf = Config()

import warnings

warnings.filterwarnings('ignore')

with open(conf.rf_model_path,'rb') as f:
    model = pickle.load(f)

with open(conf.tfidf_model_save_path,'rb') as f:
    tfidf = pickle.load(f)

def predict_fun(data):
    words = ' '.join(jieba.lcut(data['text'])[:30])
    features = tfidf.transform([words])

    y_pred_id = model.predict(features)[0]
    id2class = {i: line.strip() for i,line in enumerate(open(conf.class_doc_path,encoding='utf-8'))}

    y_pred = id2class[y_pred_id]

    data['pred_class'] = y_pred

    return data

if __name__ == '__main__':
    data = {'text':'那等你晋安帝哦'}

    result = predict_fun(data)

    print(result)