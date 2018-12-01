"""
MNIST datasets demo for gcforest
Usage:
    define the model within scripts:
        python examples/demo_mnist.py
    get config from json file:
        python examples/demo_mnist.py --model examples/demo_mnist-gc.json
        python examples/demo_mnist.py --model examples/demo_mnist-ca.json
"""
import sklearn
from sklearn import datasets
import argparse
import numpy as np
import sys
from keras.datasets import mnist
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, "lib")

from gcforest.gcforest import GCForest
from gcforest.utils.config_utils import load_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", type=str, default=None, help="gcfoest Net Model File")
    args = parser.parse_args()
    return args


def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 10
    ca_config["estimators"] = []
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 10, "max_depth": 5,
         "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config


def unsupervised_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 10
    ca_config["estimators"] = []
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomTreesEmbedding",
                                    "n_estimators": 10, "n_jobs": -1})
    config["cascade"] = ca_config
    return config


if __name__ == "__main__":
    args = parse_args()
    """
    if args.model is None:
        config = get_toy_config()
    else:
        config = load_json(args.model)
    """
    #digits = datasets.load_digits(6)
    #X = digits.data
    #print(X.shape)
    config = unsupervised_toy_config()
    gc = GCForest(config)
    # If the model you use cost too much memory for you.
    # You can use these methods to force gcforest not keeping model in memory
    # gc.set_keep_model_in_mem(False), default is TRUE.

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # X_train, y_train = X_train[:2000], y_train[:2000]
    X_train = X_train.reshape(X_train.shape[0], 28*28)
    print(X_train.shape)
    """
    X_train_flatten = []
    for i in range(X_train.shape[0]):
        train_i = X_train[i, :, :]
        train_i = train_i.flatten()
        X_train_flatten.append(X_train_flatten)

    #X_train = X_train.reshape((X_train.shape[0], -1))
    print(np.array(X_train_flatten).shape)
    """
   # X_train = X_train[:, np.newaxis, :, :]
   # X_test = X_test[:, np.newaxis, :, :]



    from sklearn.ensemble import RandomTreesEmbedding
    from sklearn.mixture import GaussianMixture
    """
    hasher = RandomTreesEmbedding(n_estimators=10, random_state=0, max_depth=5, n_jobs=-1)
    X_train_transform = hasher.fit_transform(X_train)
    hasher = RandomTreesEmbedding(n_estimators=20, random_state=0, max_depth=10, n_jobs=-1)
    X_train_transform = hasher.fit_transform(X_train_transform)
    print('Embedding transformer finished')
    print(X_train_transform.shape)
    """
    gmm = GaussianMixture(n_components=10, covariance_type='full', verbose=1, random_state=0).fit(X_train)
    y_pred = gmm.predict(X_train)
    #print(pseduo_y_train)
    acc = accuracy_score(y_train, y_pred)
    print("Test Accuracy of Other classifier using gcforest's X_encode = {:.2f} %".format(acc * 100))

