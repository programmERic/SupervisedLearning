
'''
Author: Eric Weispfenning
        eweispfenning3
'''

import numpy as np
SEED=np.random.seed(12)
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, validation_curve, learning_curve
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt


'''
Model Object
'''
class Model:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y.ravel()

    def train_test_split(self, test_size=0.4):
        self.X_test, self.X_train, self.y_test, self.y_train = train_test_split(self.X, self.y, test_size=test_size, random_state=SEED)

    def fit(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X):
        return self.model.predict(X)


    def cross_validation_curve(self, param_name, param_range):
        
        train_scores, test_scores = validation_curve(self.model,
                                                     self.X,
                                                     self.y,
                                                     param_name=param_name,
                                                     param_range=param_range)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)


        return train_scores_mean, train_scores_std, test_scores_mean, test_scores_std
    
    def plot_cross_validation_curve(self, param_name, param_range, data_name):
        
        train_scores_mean, train_scores_std, test_scores_mean, test_scores_std = self.cross_validation_curve(param_name=param_name, param_range=param_range)

        plt.figure()
        plt.title("Validation Curve with " + self.name)
        plt.xlabel(param_name)
        plt.ylabel("Score")
        plt.ylim(0.0, 1.1)
        lw = 2
        plt.semilogx(param_range, train_scores_mean, label="Training score",
                     color="darkorange", lw=lw)
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                     color="navy", lw=lw)
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color="navy", lw=lw)
        plt.legend(loc="best")
        plt.savefig(self.save_loc+self.name+data_name+"cross_validation.png")
        
    def plot_learning_curve(self, data_name, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
        
        estimator = self.model

        plt.figure()
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].set_title(self.name)
        axes[0].set_ylim((0., 1.1))
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(self.model, self.X, self.y, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes,
                           return_times=True)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

        plt.savefig(self.save_loc+data_name+self.name+"_learning_curve.png")
    

'''
Neural Network
'''
class NeuralNetwork(Model):
    def __init__(self, X, y, nn_args={}):
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        self.dtree = Model.__init__(self, MLPClassifier(random_state=SEED, **nn_args), X_scaled, y)
        self.name = "Neural Network"
        self.save_loc = "nn_figs/"

'''
Support Vector Machine
'''
class SVM(Model):
    def __init__(self, X, y, svm_args={}):
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        self.svm = Model.__init__(self, SVC(probability=True, random_state=SEED, **svm_args), X, y)
        self.name = "Support Vector Machine"
        self.save_loc = "svm_figs/"
        
'''
k Nearest Neighbor
'''
class KNN(Model):
    def __init__(self, X, y, knn_args={}):
        self.knn = Model.__init__(self, KNeighborsClassifier(**knn_args), X, y)
        self.name = "k-Nearest Neighbor"
        self.save_loc = "knn_figs/"

'''
Decision Tree
'''
class DTree(Model):
    def __init__(self, X, y, dtree_args={}):
        self.dtree = Model.__init__(self, DecisionTreeClassifier(random_state=SEED, **dtree_args), X, y)
        self.name = "Decision Tree"
        self.save_loc = "dtree_figs/"


'''
Boosting
'''
class Boost(Model):
    def __init__(self, X, y, boost_args):
        self.boost = Model.__init__(self, AdaBoostClassifier(random_state=SEED, **boost_args), X, y)
        self.name = "Boosting"
        self.save_loc = "boost_figs/"        

'''
Decision Tree Pruning Experiment
'''
def test_dtree_pruning_experiment(X, y, data_name):

    dtree = DTree(X, y)
    dtree.plot_learning_curve(data_name)

    dtree.train_test_split(test_size=0.65)
    dtree.fit()
    y_train_predict = dtree.predict(dtree.X_train)
    y_test_predict = dtree.predict(dtree.X_test)

    print()
    print("Decision Tree Accuracy")
    print("Training Accuracy: ", accuracy_score(y_train_predict, dtree.y_train))
    print("Testing Accuracy: ", accuracy_score(y_test_predict, dtree.y_test))


    pruned_tree = DTree(X, y, {'max_depth':5 ,'max_leaf_nodes':int(X.shape[1]*1/2), 'min_samples_split':2})
    pruned_tree.plot_learning_curve(data_name+'_pruned')

    pruned_tree.train_test_split(test_size=0.65)
    pruned_tree.fit()
    y_train_predict = pruned_tree.predict(pruned_tree.X_train)
    y_test_predict = pruned_tree.predict(pruned_tree.X_test)

    print()
    print("Pruned Tree Accuracy")
    print("Training Accuracy: ", accuracy_score(y_train_predict, pruned_tree.y_train))
    print("Testing Accuracy: ", accuracy_score(y_test_predict, pruned_tree.y_test))

    pruned_tree.plot_cross_validation_curve(param_name='max_leaf_nodes', param_range=range(1,X.shape[1]), data_name=data_name)

'''
SVM experiment
'''
def test_svm_kernel_experiment(X, y, data_name):

    poly_svm = SVM(X, y, {'kernel':'poly', 'degree':3, 'class_weight':'balanced', 'gamma':0.01})
    poly_svm.plot_learning_curve(data_name+'_poly_')
    
    #poly_svm.plot_cross_validation_curve(param_name='gamma', param_range=[0.1], data_name=data_name+'poly_gamma')

    poly_svm.train_test_split(test_size=0.4)
    poly_svm.fit()
    y_train_predict = poly_svm.predict(poly_svm.X_train)
    y_test_predict = poly_svm.predict(poly_svm.X_test)
    print()
    print("SVM Accuracy (kernel: Polynomial)")
    print("Training Accuracy: ", accuracy_score(y_train_predict, poly_svm.y_train))
    print("Testing Accuracy: ", accuracy_score(y_test_predict, poly_svm.y_test))


    rbf_svm = SVM(X, y, {'kernel':'rbf', 'gamma':'scale', 'class_weight':'balanced', 'gamma':0.01})
    rbf_svm.plot_learning_curve(data_name+'_rbf_')

    #rbf_svm.plot_cross_validation_curve(param_name='gamma', param_range=[0.1], data_name=data_name+'rbf_gamma')
    
    rbf_svm.train_test_split(test_size=0.4)
    rbf_svm.fit()
    y_train_predict = rbf_svm.predict(rbf_svm.X_train)
    y_test_predict = rbf_svm.predict(rbf_svm.X_test)
    print()
    print("SVM Accuracy (kernel: Radial Basis Function)")
    print("Training Accuracy: ", accuracy_score(y_train_predict, rbf_svm.y_train))
    print("Testing Accuracy: ", accuracy_score(y_test_predict, rbf_svm.y_test))

'''
kNN experiment
'''
def test_knn_experiment_heart_data():

    data_name = 'heart_'
    X, y = heart_failure_data()
    
    knn = KNN(X, y, {'n_neighbors':2, 'p':1})
    knn.plot_learning_curve(data_name)
    knn.plot_cross_validation_curve(param_name='n_neighbors', param_range=range(1,16), data_name=data_name)
    
    #knn.plot_cross_validation_curve(param_name='n_neighbors', param_range=range(1,X.shape[0]//5), data_name=data_name)

    knn.train_test_split(test_size=0.65)
    knn.fit()
    y_train_predict = knn.predict(knn.X_train)
    y_test_predict = knn.predict(knn.X_test)
    print()
    print("k-Nearest Neighbors")
    print("Training Accuracy: ", f1_score(y_train_predict, knn.y_train))
    print("Testing Accuracy: ", f1_score(y_test_predict, knn.y_test))

    
    
'''
Boosting experiment
'''
def test_boosting_experiment(X, y, data_name):

    pruned_tree = DTree(X, y, {'max_depth':5 ,'max_leaf_nodes':int(X.shape[1]*1/2), 'min_samples_split':2})

    boost1 = Boost(X, y, {'base_estimator': pruned_tree.model, 'learning_rate':3, 'n_estimators':int(X.shape[1]*1/2)})
    boost1.plot_learning_curve(data_name+'boost10nest100')

    boost1.plot_cross_validation_curve(param_name='learning_rate', param_range=range(1,15), data_name=data_name) 

'''
Neural Network experiment
'''
def test_neural_network_experiment_heart_data():

    data_name = 'heart_'
    X, y = heart_failure_data()
    
    nn = NeuralNetwork(X, y, {'hidden_layer_sizes': (3,3), 'max_iter':2000})
    nn.plot_cross_validation_curve(param_name='alpha', param_range=[0.1, 0.25, 0.5], data_name=data_name)
    nn.plot_learning_curve(data_name+'_iter33')

    nn.train_test_split(test_size=0.65)
    nn.fit()
    y_train_predict = nn.predict(nn.X_train)
    y_test_predict = nn.predict(nn.X_test)
    print()
    print("Neural Network")
    print("Training Accuracy: ", f1_score(y_train_predict, nn.y_train))
    print("Testing Accuracy: ", f1_score(y_test_predict, nn.y_test))
    
def test_neural_network_experiment_digit_data():

    data_name = 'digit_'
    nums = load_digits()
    X, y = nums.data, nums.target
    
    nn = NeuralNetwork(X, y, {'activation': 'tanh', 'hidden_layer_sizes': (3,3), 'max_iter':1000})
    nn.plot_learning_curve(data_name+'_iter2_')

    nn.train_test_split(test_size=0.65)
    nn.fit()
    y_train_predict = nn.predict(nn.X_train)
    y_test_predict = nn.predict(nn.X_test)
    print()
    print("Neural Network")
    print("Training Accuracy: ", accuracy_score(y_train_predict, nn.y_train))
    print("Testing Accuracy: ", accuracy_score(y_test_predict, nn.y_test))

    nn.plot_cross_validation_curve(param_name='alpha', param_range=[0.1, 0.25, 0.5], data_name=data_name)
    

def test_models_parameters(X, y, data_name):
    
    knn = KNN(X, y)
    knn.plot_learning_curve(data_name)
    knn.plot_cross_validation_curve(param_name='n_neighbors', param_range=range(1,16), data_name=data_name)
    
    nn = NeuralNetwork(X, y)
    nn.plot_learning_curve(data_name)
    nn.plot_cross_validation_curve(param_name='learning_rate_init', param_range=np.linspace(.001, 0.025, 10), data_name=data_name)
    
    svm = SVM(X, y)
    svm.plot_learning_curve(data_name)
    svm.plot_cross_validation_curve(param_name='degree', param_range=range(1,4), data_name=data_name)
    
    dtree = DTree(X, y)
    dtree.plot_learning_curve(data_name)
    dtree.plot_cross_validation_curve(param_name='max_depth', param_range=range(1,16), data_name=data_name)
    
    boost = Boost(X, y)
    boost.plot_learning_curve(data_name)
    boost.plot_cross_validation_curve(param_name='n_estimators', param_range=range(1,25), data_name=data_name)    


def heart_failure_data(filename='heart_failure_clinical_records_dataset.csv'):
    '''
    Heart Disease
    '''
    heart_failure = np.genfromtxt(filename, delimiter=',')

    heart_failure = heart_failure[1:, :]

    heart_X = heart_failure[:, :-1]
    heart_y = np.reshape(heart_failure[:, -1], (-1, 1))
    
    return heart_X, heart_y

def wall_clock_fit_time_digits():

    nums = load_digits()
    X, y = nums.data, nums.target
    test_size = 0.66
    
    print("Wall Clock Experiment using Digits data.")
    start = time.time()
    nn = NeuralNetwork(X, y, {'hidden_layer_sizes': (3,3), 'max_iter':2000})
    nn.train_test_split(test_size=test_size)
    nn.fit()
    finish = time.time()
    print("Neural network fit time: ", finish-start, "sec.")
    start = time.time()
    knn = KNN(X, y, {'n_neighbors':2, 'p':1})
    knn.train_test_split(test_size=test_size)
    knn.fit()
    finish = time.time()
    print("k-Nearest Neighbors fit time: ", finish-start, "sec.")
    start = time.time()
    poly_svm = SVM(X, y, {'kernel':'poly', 'degree':3, 'class_weight':'balanced'})
    poly_svm.train_test_split(test_size=test_size)
    poly_svm.fit()
    finish = time.time()
    print("SVM (polynomial kernel) fit time: ", finish-start, "sec.")
    start = time.time()
    pruned_tree = DTree(X, y, {'max_depth':5 ,'max_leaf_nodes':int(X.shape[1]*1/2), 'min_samples_split':2})
    pruned_tree.train_test_split(test_size=test_size)
    pruned_tree.fit()
    finish = time.time()
    print("Decision Tree fit time: ", finish-start, "sec.")
    start = time.time()
    boost = Boost(X, y, {'base_estimator': pruned_tree.model, 'learning_rate':3, 'n_estimators':int(X.shape[1]*1/2)})
    boost.train_test_split(test_size=test_size)
    boost.fit()
    finish = time.time()
    print("Boosting fit time: ", finish-start, "sec.")
    print()


def wall_clock_fit_time_heart():

    X, y = heart_failure_data()
    test_size = 0.66
    
    print("Wall Clock Experiment using Heart Failure data.")
    start = time.time()
    nn = NeuralNetwork(X, y, {'hidden_layer_sizes': (3,3), 'max_iter':2000})
    nn.train_test_split(test_size=test_size)
    nn.fit()
    finish = time.time()
    print("Neural network fit time: ", finish-start, "sec.")
    start = time.time()
    knn = KNN(X, y, {'n_neighbors':2, 'p':1})
    knn.train_test_split(test_size=test_size)
    knn.fit()
    finish = time.time()
    print("k-Nearest Neighbors fit time: ", finish-start, "sec.")
    start = time.time()
    poly_svm = SVM(X, y, {'kernel':'poly', 'degree':3, 'class_weight':'balanced'})
    poly_svm.train_test_split(test_size=test_size)
    poly_svm.fit()
    finish = time.time()
    print("SVM (polynomial kernel) fit time: ", finish-start, "sec.")
    start = time.time()
    pruned_tree = DTree(X, y, {'max_depth':5 ,'max_leaf_nodes':int(X.shape[1]*1/2), 'min_samples_split':2})
    pruned_tree.train_test_split(test_size=test_size)
    pruned_tree.fit()
    finish = time.time()
    print("Decision Tree fit time: ", finish-start, "sec.")
    start = time.time()
    boost = Boost(X, y, {'base_estimator': pruned_tree.model, 'learning_rate':3, 'n_estimators':int(X.shape[1]*1/2)})
    boost.train_test_split(test_size=test_size)
    boost.fit()
    finish = time.time()
    print("Boosting fit time: ", finish-start, "sec.")
    print()

    
def performance_digits():

    nums = load_digits()
    X, y = nums.data, nums.target
    test_size = 0.66

    nn = NeuralNetwork(X, y, {'hidden_layer_sizes': (3,3), 'max_iter':2000})
    nn.train_test_split(test_size=test_size)
    nn.fit()
    y_train_predict_nn = nn.predict(nn.X_train)
    y_test_predict_nn = nn.predict(nn.X_test)
    knn = KNN(X, y, {'n_neighbors':2, 'p':1})
    knn.train_test_split(test_size=test_size)
    knn.fit()
    y_train_predict_knn = knn.predict(knn.X_train)
    y_test_predict_knn = knn.predict(knn.X_test)
    poly_svm = SVM(X, y, {'kernel':'poly', 'degree':3, 'class_weight':'balanced'})
    poly_svm.train_test_split(test_size=test_size)
    poly_svm.fit()
    y_train_predict_poly_svm = poly_svm.predict(poly_svm.X_train)
    y_test_predict_poly_svm = poly_svm.predict(poly_svm.X_test)
    pruned_tree = DTree(X, y, {'max_depth':5 ,'max_leaf_nodes':int(X.shape[1]*1/2), 'min_samples_split':2})
    pruned_tree.train_test_split(test_size=test_size)
    pruned_tree.fit()
    y_train_predict_pruned = pruned_tree.predict(pruned_tree.X_train)
    y_test_predict_pruned = pruned_tree.predict(pruned_tree.X_test)
    boost = Boost(X, y, {'base_estimator': pruned_tree.model, 'learning_rate':3, 'n_estimators':int(X.shape[1]*1/2)})
    boost.train_test_split(test_size=test_size)
    boost.fit()
    y_train_predict_boost = boost.predict(boost.X_train)
    y_test_predict_boost = boost.predict(boost.X_test)

    print("Accuracy Score using Digits data.")
    print()
    print("Neural Network      Training Accuracy: ", accuracy_score(y_train_predict_nn, nn.y_train))
    print("                    Testing Accuracy : ", accuracy_score(y_test_predict_nn, nn.y_test))
    print()
    print("k-Nearest Neighbor  Training Accuracy: ", accuracy_score(y_train_predict_knn, knn.y_train))
    print("                    Testing Accuracy : ", accuracy_score(y_test_predict_knn, knn.y_test))
    print()
    print("SVM (k: polynomial) Training Accuracy: ", accuracy_score(y_train_predict_poly_svm, poly_svm.y_train))
    print("                    Testing Accuracy : ", accuracy_score(y_test_predict_poly_svm, poly_svm.y_test))
    print()
    print("Decision Tree       Training Accuracy: ", accuracy_score(y_train_predict_pruned, pruned_tree.y_train))
    print("                    Testing Accuracy : ", accuracy_score(y_test_predict_pruned, pruned_tree.y_test))
    print()
    print("Boosting            Training Accuracy: ", accuracy_score(y_train_predict_boost, boost.y_train))
    print("                    Testing Accuracy : ", accuracy_score(y_test_predict_boost, boost.y_test))
    print()

def performance_heart():

    X, y = heart_failure_data()
    test_size = 0.66

    nn = NeuralNetwork(X, y, {'hidden_layer_sizes': (3,3), 'max_iter':2000})
    nn.train_test_split(test_size=test_size)
    nn.fit()
    y_train_predict_nn = nn.predict(nn.X_train)
    y_test_predict_nn = nn.predict(nn.X_test)
    knn = KNN(X, y, {'n_neighbors':2, 'p':1})
    knn.train_test_split(test_size=test_size)
    knn.fit()
    y_train_predict_knn = knn.predict(knn.X_train)
    y_test_predict_knn = knn.predict(knn.X_test)
    poly_svm = SVM(X, y, {'kernel':'poly', 'degree':3, 'class_weight':'balanced'})
    poly_svm.train_test_split(test_size=test_size)
    poly_svm.fit()
    y_train_predict_poly_svm = poly_svm.predict(poly_svm.X_train)
    y_test_predict_poly_svm = poly_svm.predict(poly_svm.X_test)
    pruned_tree = DTree(X, y, {'max_depth':5 ,'max_leaf_nodes':int(X.shape[1]*1/2), 'min_samples_split':2})
    pruned_tree.train_test_split(test_size=test_size)
    pruned_tree.fit()
    y_train_predict_pruned = pruned_tree.predict(pruned_tree.X_train)
    y_test_predict_pruned = pruned_tree.predict(pruned_tree.X_test)
    boost = Boost(X, y, {'base_estimator': pruned_tree.model, 'learning_rate':3, 'n_estimators':int(X.shape[1]*1/2)})
    boost.train_test_split(test_size=test_size)
    boost.fit()
    y_train_predict_boost = boost.predict(boost.X_train)
    y_test_predict_boost = boost.predict(boost.X_test)

    print("Accuracy Score using Heart Disease data.")
    print()
    print("Neural Network      Training Accuracy: ", f1_score(y_train_predict_nn, nn.y_train))
    print("                    Testing Accuracy : ", f1_score(y_test_predict_nn, nn.y_test))
    print()
    print("k-Nearest Neighbor  Training Accuracy: ", f1_score(y_train_predict_knn, knn.y_train))
    print("                    Testing Accuracy : ", f1_score(y_test_predict_knn, knn.y_test))
    print()
    print("SVM (k: polynomial) Training Accuracy: ", f1_score(y_train_predict_poly_svm, poly_svm.y_train))
    print("                    Testing Accuracy : ", f1_score(y_test_predict_poly_svm, poly_svm.y_test))
    print()
    print("Decision Tree       Training Accuracy: ", f1_score(y_train_predict_pruned, pruned_tree.y_train))
    print("                    Testing Accuracy : ", f1_score(y_test_predict_pruned, pruned_tree.y_test))
    print()
    print("Boosting            Training Accuracy: ", f1_score(y_train_predict_boost, boost.y_train))
    print("                    Testing Accuracy : ", f1_score(y_test_predict_boost, boost.y_test))
    print()
    
def model_balance(y):
    print(np.bincount(y.astype(np.int32)[:,0])/y.shape[0])


def test_all_models():

    performance_digits()
    performance_heart()
    wall_clock_fit_time_digits()
    wall_clock_fit_time_heart()


if __name__ == "__main__":
    test_all_models()



    
