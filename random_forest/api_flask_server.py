from flask import Flask,request,jsonify
from rf_predict_fun import predict_fun

import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json()
    print(type(data))
    print(data)
    result = predict_fun(data)


    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8008,debug=True)