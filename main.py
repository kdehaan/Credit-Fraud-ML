import numpy as np
import csv
import sys
# import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn import svm
from sklearn import metrics
from sklearn import dummy
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt
from sklearn import preprocessing



def readCSV():
    
    with open('hotel_bookings.csv') as f:
        data = list(csv.reader(f, delimiter=","))

    np.random.seed(314)

    titles = data[0]
    data = np.array(data[1:])
    

    end_idx = data.shape[1]-1
    for i in range(end_idx+1):
        try:
            data[:,i] = data[:,i].astype(np.float64)
        except:
            le = preprocessing.LabelEncoder()
            le.fit(data[:,i])
            data[:,i] = le.transform(data[:,i])
        
    data = data.astype(np.float64)

    print("data", data[:3])

    x_data, t_data = data[:, 0:end_idx], data[:, end_idx]
    
    # x_val, t_val = val_data[:, 0:end_idx], val_data[:, end_idx]
    # x_test, t_test = test_data[:, 0:end_idx], test_data[:, end_idx]

    x_train, x_test, t_train, t_test = train_test_split(x_data, t_data, test_size=0.2, random_state=34)
    print(np.unique(t_test, return_counts=True))

    x_train, x_val, t_train, t_val = train_test_split(x_train, t_train, test_size=0.2, random_state=43)
    print(np.unique(t_val, return_counts=True))
    print(np.unique(t_train, return_counts=True))

    # print("sample", x_data[:3], t_data[:3])
    return x_train, t_train, x_test, t_test, x_val, t_val, titles


def main():
    search = False
    all_models = True
    subset = []
    if len(sys.argv) > 0:
        if "-s" in sys.argv:
            search = True
        if "-notall" in sys.argv:
            all_models = False
            if "-knn" in sys.argv:
                subset.append("knn")
            if "-sgd" in sys.argv:
                subset.append("sgd")
            if "-mlp" in sys.argv:
                subset.append("mlp")
            

    
    x_train, t_train, x_test, t_test, x_val, t_val, _ = readCSV()

    # svm_classifier = get_svm(x_train, t_train, x_val, t_val)
    # print("SVM tested at", validate_cross(svm_classifier, x_test, t_test))

    if all_models or 'sgd' in subset:    
        sgd_classifier = get_sgd(x_train, t_train, x_val, t_val, search)
        print("SGD tested at", validate_cross(sgd_classifier, x_test, t_test))

    if all_models or 'mlp' in subset:   
        mlp_classifier = get_mlp(x_train, t_train, x_val, t_val, search)
        print("MLP tested at", validate_cross(mlp_classifier, x_test, t_test))

    majority_guess = get_majority(x_train, t_train)
    uniform_guess = get_uniform(x_train, t_train)

    print("Majority tested at", validate_cross(majority_guess, x_test, t_test))
    print("Uniform tested at", validate_cross(uniform_guess, x_test, t_test))



def param_sel(x, y, model, params):
    print("Running GridSearch on hyperparameters:", params)
    grid_search = GridSearchCV( model, param_grid=params, n_jobs=2, cv=5, scoring="roc_auc")
    grid_search.fit(x, y)
    grid_search.best_params_
    return grid_search.best_params_


def get_knn(x_train, t_train, x_val, t_val, search=False):
    if search:
        knn_params = param_sel(x_train, t_train, SGDClassifier(max_iter=2000), {
            'alpha': [ 0.001, 0.006, 0.01, 0.06, 0.1, 0.6, 1],
            'loss': ['hinge', 'log', 'squared_hinge', 'modified_huber'],
            'penalty': ['l2'] })
    else:
        knn_params = {'alpha': 0.1, 'loss': 'hinge', 'penalty': 'l2'}

    print("params", knn_params)
    sgd_classifier = SGDClassifier(**knn_params, max_iter=2000)
    sgd_classifier.fit(x_train, t_train)
    print("sgd validated at", validate_cross(sgd_classifier, x_val, t_val))
    return sgd_classifier

def get_sgd(x_train, t_train, x_val, t_val, search=False):
    # {'alpha': 0.1, 'loss': 'hinge', 'penalty': 'l2'}
    # params {'alpha': 0.1, 'loss': 'squared_hinge', 'penalty': 'l2'}
    # sgd validated at (array([0.83904737, 0.77597488, 0.63281863, 0.64005236, 0.78926702]), 0.5431523356769534)
    # SGD tested at (array([0.5211474 , 0.75460637, 0.42106365, 0.84335079, 0.82848168]), 0.5295225644352521)
    if search:
        sgd_params = param_sel(x_train, t_train, SGDClassifier(max_iter=2000), {
            'alpha': [ 0.001, 0.006, 0.01, 0.06, 0.1, 0.6, 1],
            'loss': ['hinge', 'log', 'squared_hinge', 'modified_huber'],
            'penalty': ['l2'] })
    else:
        sgd_params = {'alpha': 0.1, 'loss': 'hinge', 'penalty': 'l2'}

    print("params", sgd_params)
    sgd_classifier = SGDClassifier(**sgd_params, max_iter=2000)
    sgd_classifier.fit(x_train, t_train)
    print("sgd validated at", validate_cross(sgd_classifier, x_val, t_val))
    return sgd_classifier


def get_mlp(x_train, t_train, x_val, t_val, search=False):
    # {'activation': 'relu', 'alpha': 0.1, 'learning_rate': 'constant', 'solver': 'adam'}
    # {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': (100,), 'alpha': 0.06, 'activation': 'tanh'}
    # mlp validated at (array([0.8940068 , 0.78879874, 0.71866004, 0.99057592, 0.74764398]), 0.9286105369755633)
    # MLP tested at (array([0.72152429, 0.7118928 , 0.91457286, 0.71602094, 0.70136126]), 0.9242268552514312)
    # mlp validated at (array([0.8940068 , 0.78879874, 0.71866004, 0.99057592, 0.74764398]), 0.9833533999895304)
    # MLP tested at (array([0.72152429, 0.7118928 , 0.91457286, 0.71602094, 0.70136126]), 0.9823687075969512)

    # {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (100,), 'learning_rate': 'adaptive', 'solver': 'adam'}
    # mlp validated at (array([0.98953154, 0.99188694, 0.99188694, 0.98848168, 0.98612565]), 0.9537539956508365)
    # MLP tested at (array([0.98680905, 0.98848409, 0.98911223, 0.98910995, 0.99036649]), 0.9467248198442366)
    if search:

        mlp_params = param_sel(x_train, t_train, MLPClassifier(max_iter=1000), {
            'alpha': [ 0.01, 0.06, 0.1, 0.6],
            'hidden_layer_sizes': [(20,20,10), (100,)],
            'activation': ['relu', 'tanh'],
            'solver': ['sgd', 'adam'],
            'learning_rate': ['constant', 'adaptive'], })
    else:
        mlp_params = {'activation': 'relu', 'alpha': 0.01, 'learning_rate': 'adaptive', 'solver': 'adam', 'hidden_layer_sizes': (100,)}

    print("MLP params:", mlp_params)
    mlp_classifier = MLPClassifier(**mlp_params, max_iter=6000)
    mlp_classifier.fit(x_train, t_train)
    print("mlp validated at", validate_cross(mlp_classifier, x_val, t_val))
    return mlp_classifier

def get_majority(x_train, t_train):
    majority_guess = dummy.DummyClassifier(strategy="most_frequent")
    majority_guess.fit(x_train, t_train)
    return majority_guess

def get_uniform(x_train, t_train):
    uniform_guess = dummy.DummyClassifier(strategy="uniform")
    uniform_guess.fit(x_train, t_train)
    return uniform_guess

# def validate_AUPRC(classifier, input, output):
#     predicted = classifier.predict(input)
#     auprc = metrics.average_precision_score(output, predicted)
#     return auprc

def validate_cross(classifier, x, y):
    return (cross_val_score(classifier, x, y, cv=5), r2_score(y, classifier.predict(x)), classifier.score(x, y))

if __name__ == "__main__":
    main()
    
