import numpy as np
import csv
import sys
# import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn import svm
from sklearn import metrics
from sklearn import dummy
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, roc_curve, auc, precision_recall_curve
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
    # print("data", data[:3])

    x_data, t_data = data[:, 0:end_idx], data[:, end_idx]
    # x_val, t_val = val_data[:, 0:end_idx], val_data[:, end_idx]
    # x_test, t_test = test_data[:, 0:end_idx], test_data[:, end_idx]

    x_train, x_test, t_train, t_test = train_test_split(x_data, t_data, test_size=0.2, random_state=34)
    # print(np.unique(t_test, return_counts=True))

    x_train, x_val, t_train, t_val = train_test_split(x_train, t_train, test_size=0.2, random_state=43)
    # print(np.unique(t_val, return_counts=True))
    # print(np.unique(t_train, return_counts=True))

    # print("sample", x_data[:3], t_data[:3])
    return x_train, t_train, x_test, t_test, x_val, t_val, titles


def main():
    search = False
    all_models = True
    save_plots = False
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
        if "-plot" in sys.argv:
            save_plots = True
            

    
    x_train, t_train, x_test, t_test, x_val, t_val, _ = readCSV()



    if all_models or 'knn' in subset:
        knn_classifier = get_knn(x_train, t_train, x_val, t_val, search)
        print("KNN tested at", validate(knn_classifier, x_test, t_test))
        if (save_plots):
            plot_auc("KNN", t_test, knn_classifier.predict(x_test))
            plot_prec_rec("KNN", t_test, knn_classifier.predict(x_test))
            # plot_learning_curve(knn_classifier, "KNN", x_train, t_train)

    if all_models or 'sgd' in subset:    
        sgd_classifier = get_sgd(x_train, t_train, x_val, t_val, search)
        print("SGD tested at", validate(sgd_classifier, x_test, t_test))
        if (save_plots):
            plot_auc("SGD", t_test, sgd_classifier.predict(x_test))
            plot_prec_rec("SGD", t_test, sgd_classifier.predict(x_test))
            # plot_learning_curve(sgd_classifier, "SGD", x_train, t_train)

    if all_models or 'mlp' in subset:   
        mlp_classifier = get_mlp(x_train, t_train, x_val, t_val, search)
        print("MLP tested at", validate(mlp_classifier, x_test, t_test))
        if (save_plots):
            plot_auc("MLP", t_test, mlp_classifier.predict(x_test))
            plot_prec_rec("MLP", t_test, mlp_classifier.predict(x_test))
            # plot_learning_curve(mlp_classifier, "MLP", x_train, t_train)

    majority_guess = get_majority(x_train, t_train)
    uniform_guess = get_uniform(x_train, t_train)

    print("Majority tested at", validate(majority_guess, x_test, t_test))
    print("Uniform tested at", validate(uniform_guess, x_test, t_test))


def plot_auc(name, y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_val = roc_auc_score(y_true, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % auc_val)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f'{name} Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f'{name}_roc_auc.png')

def plot_prec_rec(name, y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auc_val = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label='Area under the curve: %0.2f' % auc_val)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f'{name} Precision Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'{name}_prec_rec.png')

# adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, name, x, y, train_sizes=np.linspace(0.1, 1.0, 5)):
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, x, y, train_sizes=train_sizes, return_times=True)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    # fit_times_std = np.std(fit_times, axis=1)

    plt.figure()
    plt.title(f'{name} Learning Curves')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='b')
    plt.plot(train_sizes, train_scores_mean, '-o', label="Training Score", color='r')
    plt.plot(train_sizes, test_scores_mean, '-o', label="Cross-Validation Score", color='b')
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.savefig(f'{name}_learning_curve.png')

    plt.figure()
    plt.title(f'{name} Model Performance')
    plt.fill_between(fit_times_mean, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1 )
    plt.plot(fit_times_mean, test_scores_mean, '-o')
    plt.xlabel("Fit Times")
    plt.ylabel("Score")
    plt.savefig(f'{name}_model_performance.png')



def param_sel(x, y, model, params):
    print("Running GridSearch on hyperparameters:", params)
    grid_search = GridSearchCV( model, param_grid=params, n_jobs=2, cv=5, scoring="roc_auc")
    grid_search.fit(x, y)
    grid_search.best_params_
    return grid_search.best_params_


def get_knn(x_train, t_train, x_val, t_val, search=False):
    # KNN params: {'algorithm': 'kd_tree', 'leaf_size': 30, 'n_neighbors': 20, 'p': 1, 'weights': 'distance'}
    # KNN tested at (array([0.88484087, 0.88107203, 0.89007538, 0.88628272, 0.89256545]), 0.6603707265070087, 0.9209732808442919)
    if search:
        knn_params = param_sel(x_train, t_train, KNeighborsClassifier(), {
            'n_neighbors': [ 3, 10, 20],
            'weights': ['uniform', 'distance'],
            'algorithm': ['ball_tree', 'kd_tree',],
            'p': [1, 2]})
    else:
        knn_params = {'algorithm': 'kd_tree', 'leaf_size': 30, 'n_neighbors': 20, 'p': 1, 'weights': 'distance'}

    
    knn_classifier = KNeighborsClassifier(**knn_params)
    knn_classifier.fit(x_train, t_train)
    print("KNN params:", knn_classifier.get_params())
    print("KNN validated at", validate(knn_classifier, x_val, t_val))
    return knn_classifier

def get_sgd(x_train, t_train, x_val, t_val, search=False):
    # {'alpha': 0.1, 'loss': 'hinge', 'penalty': 'l2'}
    # params {'alpha': 0.1, 'loss': 'squared_hinge', 'penalty': 'l2'}
    # sgd validated at (array([0.83904737, 0.77597488, 0.63281863, 0.64005236, 0.78926702]), 0.5431523356769534)
    # SGD tested at (array([0.5211474 , 0.75460637, 0.42106365, 0.84335079, 0.82848168]), 0.5295225644352521)
    if search:
        sgd_params = param_sel(x_train, t_train, SGDClassifier(max_iter=2000), {
            'alpha': [ 0.01, 0.06, 0.1, 0.6, 1],
            'loss': ['hinge', 'log', 'squared_hinge', 'modified_huber'],
            'penalty': ['l2'] })
    else:
        sgd_params = {'alpha': 0.1, 'loss': 'squared_hinge', 'penalty': 'l2'}


    sgd_classifier = SGDClassifier(**sgd_params, max_iter=2000)
    sgd_classifier.fit(x_train, t_train)
    print("SGD params:", sgd_classifier.get_params())
    print("SGD validated at", validate(sgd_classifier, x_val, t_val))
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

    mlp_classifier = MLPClassifier(**mlp_params, max_iter=6000)
    mlp_classifier.fit(x_train, t_train)
    print("MLP params:", mlp_classifier.get_params())
    print("MLP validated at", validate(mlp_classifier, x_val, t_val))
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

def validate(classifier, x, y):
    score_r2 = r2_score(y, classifier.predict(x))
    score_accuracy = classifier.score(x, y)
    score_auc = roc_auc_score(y, classifier.predict(x))
    return f'r2_score: {score_r2}, accuracy: {score_accuracy}, ROC AUC: {score_auc}'

if __name__ == "__main__":
    main()
    
