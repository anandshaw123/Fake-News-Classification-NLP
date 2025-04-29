from flask import Flask, jsonify, request, render_template
from predict import Predict
import pandas as pd
import json
from random import randrange
import codecs

app = Flask(__name__, static_folder="./public/static", template_folder="./public")

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/random', methods=['GET'])
def random():
    data = pd.read_csv("./data/fake_or_real_news_testset.csv")
    index = randrange(0, len(data)-1, 1)
    response = jsonify({'title': data.loc[index].title, 'text': data.loc[index].text})
    return response

@app.route('/predict', methods=['POST'])
def predict():
    param = jsonify((request.data).decode('utf-8'))
    param = json.loads(param.json)
    param = param['article']
    model = Predict(param)
    response = jsonify(model.predict())
    return response

if __name__ == '__main__':
    app.run()

