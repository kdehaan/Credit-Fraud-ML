import numpy as np
import csv
# import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn import dummy
import matplotlib.pyplot as plt
from sklearn import preprocessing



def readCSV():
    
    with open('hotel_bookings.csv') as f:
        data = list(csv.reader(f, delimiter=","))

    np.random.seed(314)

    titles = data[0]
    data = np.array(data[1:])
    

    end_idx = data.shape[1]-1
    for i in range(end_idx):
        le = preprocessing.LabelEncoder()
        le.fit(data[:,i])
        data[:,i] = le.transform(data[:,i])

    data = data.astype(np.float)

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
    x_train, t_train, x_test, t_test, x_val, t_val, _ = readCSV()

    # svm_classifier = get_svm(x_train, t_train, x_val, t_val)
    # print("SVM tested at", validate_cross(svm_classifier, x_test, t_test))

    sgd_classifier = get_sgd(x_train, t_train, x_val, t_val)
    print("SGD tested at", validate_cross(sgd_classifier, x_test, t_test))

    mlp_classifier = get_mlp(x_train, t_train, x_val, t_val)
    print("MLP tested at", validate_cross(mlp_classifier, x_test, t_test))

    majority_guess = get_majority(x_train, t_train)
    uniform_guess = get_uniform(x_train, t_train)

    print("Majority tested at", validate_cross(majority_guess, x_test, t_test))
    print("Uniform tested at", validate_cross(uniform_guess, x_test, t_test))


def get_svm(x_train, t_train, x_val, t_val):
    svm_classifier = svm.SVC()
    svm_classifier.fit(x_train, t_train)
    print("svm validated at", validate_cross(svm_classifier, x_val, t_val))
    return svm_classifier

def get_sgd(x_train, t_train, x_val, t_val):
    sgd_classifier = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)
    sgd_classifier.fit(x_train, t_train)
    print("sgd validated at", validate_cross(sgd_classifier, x_val, t_val))
    return sgd_classifier

def get_mlp(x_train, t_train, x_val, t_val):
    mlp_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
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
    return cross_val_score(classifier, x, y, cv=5)

if __name__ == "__main__":
    main()
    
