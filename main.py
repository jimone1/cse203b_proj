from sklearn import svm
import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

FASHION_MNIST="Fashion-MNIST" # https://github.com/zalandoresearch/fashion-mnist 10 classes
BREAST_CANCER="Breast Cancer Wisconsin (Diagnostic) Dataset" # https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic) 2 classes
IRIS_DATASET="Iris Dataset" # https://archive.ics.uci.edu/ml/datasets/iris 3 classes
WINE_DATASET="Wine Dataset" # https://archive.ics.uci.edu/ml/datasets/wine 2 classes

ALL_DATASET = {
    FASHION_MNIST,
    BREAST_CANCER,
    IRIS_DATASET,
    WINE_DATASET
}

# Load Fashion MNIST dataset
def loadDataset(datasetName):
    if (datasetName == FASHION_MNIST):
        return fetch_openml('Fashion-MNIST', version=1, cache=True)
    if (datasetName == BREAST_CANCER):
        return load_breast_cancer()
    if (datasetName == IRIS_DATASET):
        return load_iris()
    if (datasetName == WINE_DATASET):
        return load_wine()

def loadData(datasetName):
    print(f"Loading {datasetName} Data")
    dataset = loadDataset(datasetName)

    # Extract features and labels
    print(f"Labeling {datasetName} Data")
    X = np.array(dataset['data'], dtype=np.float32)
    y = np.array(dataset['target'], dtype=np.int64)
    
    classes = set(y)
    print(f"There are {len(classes)} labels: {classes}")
    return X, y


# This can be implemented later.
def preProcessing(X, y):
    return X, y

# TODO: Implement Convex optimization here!
def dataProcessing(X):
    # Applies Convex Optimization.
    # Scale features using StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(X)
    

def training(X, y):
    # Split dataset into training and testing sets
    print("Convert On Dataset")
    train_samples = len(X) // 2
    X_train, y_train = X[:train_samples], y[:train_samples]
    X_test, y_test = X[train_samples:], y[train_samples:]

    # Train SVM classifier with default hyperparameters
    print("Train On Dataset")
    clf = svm.SVC()
    clf.fit(X_train, y_train)

    # Evaluate the classifier on the test set
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print('Test accuracy:', acc)

if __name__ == '__main__':
    # Dataset Refs:
        # 1. FASHION_MNIST
        # 2. BREAST_CANCER
        # 3. IRIS_DATASET
        # 4. WINE_DATASET
    
    i = 0
    for data_set in ALL_DATASET:
        print(f"{i}: {data_set}")
        X, y = loadData(data_set)
        
        print("Data preprocessing")
        X, y = preProcessing(X, y)

        print("Train Before Data Processing")
        training(X, y)

        print("Train After Data Processing")
        X = dataProcessing(X)
        training(X, y)
        print()
        i += 1

# Convex Optimization: fit_transform 
# Performed under SVM classifier.

# Before: applied svm directly to the dataset.
# After: fit_transform on training data

# Accuracy
#         Wine data set.  Breast Cancer.    Iris
# Before      44.9 %        89.8%           33.3 %
# After       46.1 %        97.5%          33.3 %

# Data: Base        Filtered Case (Binarization, Convex Opt) Binary -> convex opt
# Convex opt -> normal datset
#     Base Acc        Improved Acc
#       30%             60%                         -> 30%
# Convex opt -> filtered dataset 
#     Base Acc        Improved Acc
#       60%             70% (best perf)             -> 10%

# Blurring -> Image Smoothing, Image Sharpening -> Binarization