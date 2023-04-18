import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from flask import Flask, jsonify, request
from keras.models import load_model

def heuristic(df):
    X = df
    X['pred'] = -1
    X['pred'][(X[0] > 2400) & (X[9] < 1200) & (X[1] <= 20)] = 5
    X['pred'][(X[9] < 900) & (X[0] < 2400) & (X[3] < 5)] = 3
    X['pred'][(X[9] < 900) & (X[0] < 2400) & (X[3] >= 5)] = 2
    X['pred'][(X[0] > 2400) & (X[3] < 100)] = 0
    X['pred'][(X[0] > 2400) & (X[3] >= 100)] = 1
    X['pred'][(X[0] > 3200)] = 6
    X['pred'][X['pred'] == 0] = 4

    return X['pred']

model_decision_tree = joblib.load("Decision_tree.joblib")
model_random_forest = joblib.load("Random_forest.joblib")
model_NN = load_model('saved_model/NN_model')

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    model = data['model']
    features = pd.DataFrame([data['features']])

    if model == 'heuristic':
        prediction = heuristic(features)
    elif model == 'base1':
        prediction = str(model_decision_tree.predict(features).flatten())
    elif model == 'base2':
        prediction = str(model_random_forest.predict(features).flatten())
    elif model == 'neural_network':
        predicted = list(model_NN.predict(features).flatten())
        prediction = str(predicted.index(max(predicted)))
    else:
        return jsonify({'prediction': 'Invalid model'})

    return jsonify({'prediction': prediction})


if __name__ == "__main__":
    app.run(debug=True)
