import streamlit as st
import requests
import time

st.title('投满分分类项目')
st.write("这是一个投满分分类项目")
text = st.text_input('请输入查询的分类文本：')
url = 'http://127.0.0.1:8000/predict'

if st.button('基于随机森林模型获取分类！'):
    start_time = time.time()
    try:
        r = requests.post(url,json = {'text':text})
        print(r.json())
        estimated_time = (time.time() - start_time) * 1000
        st.write('耗时：',estimated_time,'ms')
        st.write('预测结果',r.json()['pred_class'])
    except Exception as e:
        print('Error occurred:',e)