import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model
import seaborn as sns

pd.set_option('mode.chained_assignment', None)
plt.rcParams["figure.figsize"] = (12, 8)


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


def create_model(neurons=1, activation='relu', optimizer='adam'):
    model = Sequential()
    model.add(Dense(neurons, input_dim=54, activation=activation))
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def grid_search(X, y):
    model = KerasClassifier(build_fn=create_model, verbose=0)
    param_grid = {'neurons': [16, 32, 64, 128, 256], 'activation': ['relu', 'sigmoid', 'tanh'],
                  'optimizer': ['adam', 'sgd']}
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X, y)
    print(f"Best parameters: {grid_result.best_params_}")
    return grid_result.best_params_


def plot_history(network_history):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(network_history.history['loss'])
    plt.plot(network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(network_history.history['accuracy'])
    plt.plot(network_history.history['val_accuracy'])
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()


def create_decision_tree(depths=16):
    clf = DecisionTreeClassifier(max_depth=depths)
    clf = clf.fit(X_train, y_train)
    clf_report = classification_report(y_test, clf.predict(X_test), target_names=['0', '1', '2', '3', '4', '5', '6'],
                                       output_dict=True)
    dump(clf, 'Decision_tree.joblib')
    return clf, clf_report


def create_random_forest(depth=16):
    clf_rand = RandomForestClassifier(max_depth=depth, random_state=0)
    clf_rand.fit(X_train, y_train)
    clf_rand_report = classification_report(y_test, clf_rand.predict(X_test),
                                            target_names=['0', '1', '2', '3', '4', '5', '6'], output_dict=True)
    dump(clf_rand, 'Random_forest.joblib')
    return clf_rand, clf_rand_report


def output_NN(predicted):
    predicted.flatten()
    return predicted.index(max(predicted))


# Prepearing data to process
data1 = pd.read_csv('covtype.data', sep=",", header=None)
target = data1.iloc[:, -1].astype(int) - 1
for col in data1.columns:
    data1[col] = data1[col].astype(float)
data1['target'] = target.astype(int)

# Evaluation of accuracy for heuristic model
pred = heuristic(df)
heuristic_acc = sum(target == pred) / len(data1)

# Spliting data into 75% training 25% test.
X_train, X_test, y_train, y_test = train_test_split(data1.iloc[:, :-3], data1.target, random_state=1)

# normalizing data for neural networks.
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# After modification of grid search this line can show the best hyperparameters for model
# params = grid_search(X_scaled, y_train)

# Creating NN model with valyes predicted from grid_search.
model = Sequential()
model.add(Dense(256, input_dim=54, activation='relu'))
model.add(Dense(7, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fitting model and saving history for plotting
network_history = model.fit(X_train, y_train,
                            epochs=20, verbose=1, validation_data=(X_test, y_test))

# save model to a file
model.save('saved_model/NN_model')

# test if model can be loaded correctly
# model = load_model('saved_model/NN_model')

#clf = load('Decision_tree.joblib')
#clf_rand = load('Random_forest.joblib')

clf,clf_report=create_decision_tree()
clf_rand,clf_rand_report=create_random_forest()

# Comparision by plots, it's easy to see that NN has the best results.
plot_history(network_history)

sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
plt.title("Decision Tree", fontsize=20)

sns.heatmap(pd.DataFrame(clf_rand_report).iloc[:-1, :].T, annot=True)
plt.title("Random Forest", fontsize=20)
