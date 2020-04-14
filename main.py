import numpy as np
import csv
# import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import metrics
from sklearn import dummy
import matplotlib.pyplot as plt

END_IDX = 30

def readCSV():
    
    with open('creditcard.csv') as f:
        data = list(csv.reader(f, delimiter=","))

    np.random.seed(314)

    titles = data[0]
    data = data[1:]

    np.random.shuffle(data)

    testTrainThreshold = int(len(data)*0.8)
    trainValThreshold = int(testTrainThreshold*0.8)

    train_data, test_data = np.array(data[:testTrainThreshold], np.float), np.array(data[testTrainThreshold:], np.float)
    train_data, val_data = train_data[:trainValThreshold], train_data[trainValThreshold:]

    x_train, t_train = train_data[:, 0:END_IDX], train_data[:, END_IDX]
    x_val, t_val = val_data[:, 0:END_IDX], val_data[:, END_IDX]
    x_test, t_test = test_data[:, 0:END_IDX], test_data[:, END_IDX]

    return x_train, t_train, x_test, t_test, x_val, t_val, titles


def main():
    x_train, t_train, x_test, t_test, x_val, t_val, titles = readCSV()

    # svm_classifier = get_svm(x_train, t_train, x_val, t_val)
    # print("SVM tested at", validate_AUPRC(svm_classifier, x_test, t_test))

    # sgd_classifier = get_sgd(x_train, t_train, x_val, t_val)
    # print("SGD tested at", validate_AUPRC(sgd_classifier, x_test, t_test))

    mlp_classifier = get_mlp(x_train, t_train, x_val, t_val)
    print("MLP tested at", validate_AUPRC(mlp_classifier, x_test, t_test))

    majority_guess = get_majority(x_train, t_train)
    uniform_guess = get_uniform(x_train, t_train)

    print("Majority tested at", validate_AUPRC(majority_guess, x_test, t_test))
    print("Uniform tested at", validate_AUPRC(uniform_guess, x_test, t_test))


def get_svm(x_train, t_train, x_val, t_val):
    svm_classifier = svm.SVC()
    svm_classifier.fit(x_train, t_train)
    print("svm validated at", validate_AUPRC(svm_classifier, x_val, t_val))
    return svm_classifier

def get_sgd(x_train, t_train, x_val, t_val):
    sgd_classifier = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)
    sgd_classifier.fit(x_train, t_train)
    print("sgd validated at", validate_AUPRC(sgd_classifier, x_val, t_val))
    return sgd_classifier

def get_mlp(x_train, t_train, x_val, t_val):
    mlp_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    mlp_classifier.fit(x_train, t_train)
    print("mlp validated at", validate_AUPRC(mlp_classifier, x_val, t_val))
    return mlp_classifier

def get_majority(x_train, t_train):
    majority_guess = dummy.DummyClassifier(strategy="most_frequent")
    majority_guess.fit(x_train, t_train)
    return majority_guess

def get_uniform(x_train, t_train):
    uniform_guess = dummy.DummyClassifier(strategy="uniform")
    uniform_guess.fit(x_train, t_train)
    return uniform_guess

def validate_AUPRC(classifier, input, output):
    predicted = classifier.predict(input)
    auprc = metrics.average_precision_score(output, predicted)
    return auprc

if __name__ == "__main__":
    main()
    
