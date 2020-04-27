# Hotel Cancellations ML

Experiments with ML for predicting if hotel reservations will be cancelled

Source and info on data: https://www.kaggle.com/jessemostipak/hotel-booking-demand

How to run this code:

In the repository with python >= 3.6, type 'python main.py' or 'python3 main.py' according to your installation.

Optional command line arguments:
-s: searches for optimal hyperparameters using GridSearchCV (warning: takes a _very_ long time)
-plot: create plots for learning curves, model performance, and AUC graphs of each model (warning: takes a reasonably long time)
-notall: trains and runs only the models selected

When '-notall' is used:
-knn: trains and runs K-nearest neighbors
-sgd: trains and runs a linear classifier trained using Stochastic Gradient Descent
-mlp: trains and runs a Multi-layer Perceptron classifier
