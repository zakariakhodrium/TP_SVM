# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from ISLP import confusion_table

from svm_source import *
from sklearn import svm
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from time import time

scaler = StandardScaler()

import warnings

warnings.filterwarnings("ignore")

plt.style.use("ggplot")
# %%
###############################################################################
#               Toy dataset : 2 gaussians
###############################################################################
"""Sample n1 and n2 points from two Gaussian variables centered in mu1,
    mu2, with respective std deviations sigma1 and sigma2n1 = 200"""
np.random.seed(2)
n1 = 200
n2 = 200
mu1 = [1.0, 1.0]
mu2 = [-1.0 / 2, -1.0 / 2]
sigma1 = [0.9, 0.9]
sigma2 = [0.9, 0.9]
X1, y1 = rand_bi_gauss(n1, n2, mu1, mu2, sigma1, sigma2)

plt.show()
plt.close("all")
plt.ion()
plt.figure(1, figsize=(15, 6))
plot_2d(X1, y1)
# %%
X_train = X1[::2]
Y_train = y1[::2].astype(int)
X_test = X1[1::2]
Y_test = y1[1::2].astype(int)

# fit the model with linear kernel
clf = SVC(kernel="linear")
clf.fit(X_train, Y_train)

# predict labels for the test data base
y_pred = clf.predict(X_test)

# check your score
score = clf.score(X_test, Y_test)
print("Score : %s" % score)


# display the frontiere
def f(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf.predict(xx.reshape(1, -1))


plt.figure()
plt.title("frontiere for linear Kernel SVM")
frontiere(f, X_train, Y_train, w=None, step=50, alpha_choice=1)
# %%
# Same procedure but with a grid search
parameters = {"kernel": ["linear"], "C": list(np.linspace(0.001, 3, 200))}

clf2 = SVC()
clf_grid = GridSearchCV(clf2, parameters, n_jobs=-1)
clf_grid.fit(X_train, Y_train)

# check your score
print("The best parameters using gridsearchCV are", clf_grid.best_params_)
print("Score by using grid searchCV is: %s" % clf_grid.score(X_test, Y_test))


def f_grid(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_grid.predict(xx.reshape(1, -1))


# display the frontiere
plt.figure(figsize=(5, 5))
plt.title("Grid search frontiere")
frontiere(f_grid, X_train, Y_train, w=None, step=50, alpha_choice=1)

# %%
###############################################################################
#               Iris Dataset
###############################################################################

iris = datasets.load_iris()  # Get the data and meta data as a dictionary
X = iris.data  # get the data from the dictionary
X = scaler.fit_transform(X)  # standarize the data
y = iris.target  # get the response variables
IRIS = pd.DataFrame(X, columns=list(iris.feature_names))
IRIS["Plant"] = y
IRIS
# %%
X = X[y != 0, :2]  # Select the first two variables : sepal length (cm),sepal width (cm)
y = y[y != 0]  # Select the the targets versicolor (1) and virginica (2)
# %%
plt.show()
plt.figure(1, figsize=(16, 8))
plot_2d(X, y)
# %%
# split train test
X, y = shuffle(X, y, random_state=18)
# the correspondence between each row in X and its label in y is maintained.
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(
    X, y, test_size=0.5, random_state=47
)

# %%
###############################################################################
# fit the model with linear vs polynomial kernel
###############################################################################
# %%
# Q1 Linear kernel

# fit the model
clf_linear = SVC(kernel="linear")
clf_linear.fit(X_iris_train, y_iris_train)
print("Score of a linear classifier is", clf_linear.score(X_iris_test, y_iris_test))
# %%
parameters = {"kernel": ["linear"], "C": list(np.logspace(-3, 3, 2000))}
clf_linear_grid = GridSearchCV(clf_linear, parameters, n_jobs=-1)
clf_linear_grid.fit(X_iris_train, y_iris_train)
print("Best param using GridSearchCV is", clf_linear_grid.best_estimator_)

# %%
print(
    "Generalization score for linear kernel: %s, %s"
    % (
        clf_linear.score(X_iris_train, y_iris_train),
        clf_linear.score(X_iris_test, y_iris_test),
    )
)
confusion_table(clf_linear.predict(X_iris_test), y_iris_test)

# %%
# compute the score
print(
    "Generalization score for linear kernel using both train and test sets using gridsearchCV: %s, %s"
    % (
        clf_linear_grid.score(X_iris_train, y_iris_train),
        clf_linear_grid.score(X_iris_test, y_iris_test),
    )
)
confusion_table(clf_linear_grid.predict(X_iris_test), y_iris_test)

# %%
# Q2 polynomial kernel
Cs = list(np.logspace(-3, 3, 5))
gammas = 10.0 ** np.arange(1, 2)
degrees = np.r_[1, 2, 3]

clf_poly = SVC(kernel="poly")
clf_poly.fit(X_iris_train, y_iris_train)
print(clf_poly.score(X_iris_test, y_iris_test))

# %%
print(
    "Generalization score for polynomial kernel using both train \
    and test sets: %s, %s"
    % (
        clf_poly.score(X_iris_train, y_iris_train),
        clf_poly.score(X_iris_test, y_iris_test),
    )
)
confusion_table(clf_poly.predict(X_iris_test), y_iris_test)
# %%
parameters = {"kernel": ["poly"], "C": Cs, "gamma": gammas, "degree": degrees}
clf_poly_grid = GridSearchCV(clf_poly, parameters, n_jobs=-1)
clf_poly_grid.fit(X_iris_train, y_iris_train)
clf_poly_grid.best_estimator_

# %%
print(
    "Generalization score for polynomial kernel for train and test sets \
    using GridSearchCV: %s, %s"
    % (
        clf_poly_grid.score(X_iris_train, y_iris_train),
        clf_poly_grid.score(X_iris_test, y_iris_test),
    )
)
confusion_table(clf_poly_grid.predict(X_iris_test), y_iris_test)
# %%
# display your results using frontiere


def f_linear(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_linear_grid.predict(xx.reshape(1, -1))


def f_poly(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_poly_grid.predict(xx.reshape(1, -1))


plt.ion()
plt.figure(figsize=(18, 7))
plt.subplot(131)
plot_2d(X, y)
plt.title("iris dataset")

plt.subplot(132)
frontiere(f_linear, X, y)
plt.title("linear kernel")

plt.subplot(133)
frontiere(f_poly, X, y)

plt.title("polynomial kernel")
plt.tight_layout()
plt.draw()

# %%
###############################################################################
#               SVM GUI
###############################################################################

# please open a terminal and run python svm_gui.py
# Then, play with the applet : generate various datasets and observe the
# different classifiers you can obtain by varying the kernel


# %%
###############################################################################
#               Face Recognition Task
###############################################################################
"""
The dataset used in this example is a preprocessed excerpt
of the "Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

  _LFW: http://vis-www.cs.umass.edu/lfw/
"""

####################################################################
# Download the data and unzip; then load it as numpy arrays
lfw_people = fetch_lfw_people(
    min_faces_per_person=70,
    resize=0.4,
    color=True,
    funneled=False,
    slice_=None,
    download_if_missing=True,
)
# data_home='.'

# introspect the images arrays to find the shapes (for plotting)
images = lfw_people.images
n_samples, h, w, n_colors = images.shape

# the label to predict is the id of the person
target_names = lfw_people.target_names.tolist()

####################################################################
# Pick a pair to classify such as
# names = ['Tony Blair', 'Colin Powell']
names = ["Donald Rumsfeld", "Colin Powell"]

idx0 = lfw_people.target == target_names.index(names[0])
idx1 = lfw_people.target == target_names.index(names[1])
images = np.r_[images[idx0], images[idx1]]
n_samples = images.shape[0]
y = np.r_[np.zeros(np.sum(idx0)), np.ones(np.sum(idx1))].astype(int)

# plot a sample set of the data
plot_gallery(images, np.arange(12))
plt.show()

# %%
####################################################################
# Extract features

# features using only illuminations
X_img = (np.mean(images, axis=3)).reshape(n_samples, -1)


# Scale features
X_img -= np.mean(X_img, axis=0)
X_img /= np.std(X_img, axis=0)

# %%
####################################################################
# Split data into a half training and half test set
X_img_train, X_img_test, y_train, y_test, images_train, images_test = train_test_split(
    X_img, y, images, test_size=0.5, random_state=0
)
# X_train, X_test, y_train, y_test = \
#    train_test_split(X, y, test_size=0.5, random_state=0)

indices = np.random.permutation(X_img.shape[0])
train_idx, test_idx = indices[: X_img.shape[0] // 2], indices[X_img.shape[0] // 2 :]
X_img_train, X_img_test = X_img[train_idx, :], X_img[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]
images_train, images_test = images[train_idx, :, :, :], images[test_idx, :, :, :]

####################################################################
# Quantitative evaluation of the model quality on the test set

# %%
# Q3
print("--- Linear kernel ---")
print("Fitting the classifier to the training set")

t0 = time()

Cs = 10.0 ** np.arange(-5, 6)  # Regularization parameters to try
scores = []

# Loop over all Cs and train the model
for C in Cs:
    clf = SVC(kernel="linear", C=C)  # Create SVM with linear kernel and C
    clf.fit(X_img_train, y_train)  # Fit the model
    scores.append(clf.score(X_img_test, y_test))  # Evaluate and save the score

# Get the best C with the highest score
ind = np.argmax(scores)
best_C = Cs[ind]
print("Best C: {}".format(best_C))

# Plot the scores vs. C values
plt.figure()
plt.plot(Cs, scores)
plt.xlabel("Regularization parameter log(C)")
plt.ylabel("Test set score")
plt.xscale("log")
plt.tight_layout()
plt.show()

print("Best score: ".format(np.max(scores)))

print("Predicting the people names on the testing set")
t0 = time()

# %%
# predict labels for the X_test images with the best classifier
clf_img = SVC(kernel="linear", C=best_C)
clf_img.fit(X_img_train, y_train)
y_predicted = clf_img.predict(X_img_test)

print("done in %0.3fs" % (time() - t0))
# The chance level is the accuracy that will be reached when constantly predicting the majority class.
print("Chance level : %s" % max(np.mean(y), 1.0 - np.mean(y)))
print("Accuracy : %s" % clf_img.score(X_img_test, y_test))
# %%
confusion_table(y_predicted, y_test)
# %%
####################################################################
# Qualitative evaluation of the predictions using matplotlib
prediction_titles = [
    title(y_predicted[i], y_test[i], names) for i in range(y_predicted.shape[0])
]
plot_gallery(images_test, prediction_titles, n_row=5, n_col=5)
plt.show()


# %%
####################################################################
# Look at the coefficients
plt.figure()
plt.imshow(np.reshape(clf_img.coef_, (h, w)))
plt.show()

# %%
# Q4


def run_svm_cv(_X, _y):
    _indices = np.random.permutation(_X.shape[0])
    _train_idx, _test_idx = _indices[: _X.shape[0] // 2], _indices[_X.shape[0] // 2 :]
    _X_train, _X_test = _X[_train_idx, :], _X[_test_idx, :]
    _y_train, _y_test = _y[_train_idx], _y[_test_idx]

    _parameters = {"kernel": ["linear"], "C": list(np.logspace(-3, 3, 5))}
    _svr = svm.SVC()
    _clf_linear = GridSearchCV(_svr, _parameters, n_jobs=-1)
    _clf_linear.fit(_X_train, _y_train)

    print(
        "Generalization score for linear kernel: %s, %s \n"
        % (_clf_linear.score(_X_train, _y_train), _clf_linear.score(_X_test, _y_test))
    )


# %%


class SVM_CV:
    def __init__(self):
        self.score = None  # To store the score as an attribute

    def run_svm_cv(self, _X, _y):
        _indices = np.random.permutation(_X.shape[0])
        _train_idx, _test_idx = (
            _indices[: _X.shape[0] // 2],
            _indices[_X.shape[0] // 2 :],
        )
        _X_train, _X_test = _X[_train_idx, :], _X[_test_idx, :]
        _y_train, _y_test = _y[_train_idx], _y[_test_idx]

        _parameters = {"kernel": ["linear"], "C": list(np.logspace(-3, 3, 5))}
        _svr = svm.SVC()
        _clf_linear = GridSearchCV(_svr, _parameters, n_jobs=-1)
        _clf_linear.fit(_X_train, _y_train)

        train_score = _clf_linear.score(_X_train, _y_train)
        test_score = _clf_linear.score(_X_test, _y_test)
        self.score = {"test_score": test_score}  # Store scores in the class attribute

        return print(
            f"Generalization score for linear kernel: {train_score}, {test_score} \n"
        )


# %%
print("Score sans variable de nuisance")
clf_cv = SVM_CV()
clf_cv.run_svm_cv(X_img, y)
# %%
# On rajoute des variables de nuisances
n_features = X_img.shape[1]
sigma = 1
noise = sigma * np.random.randn(
    n_samples,
    300,
)
X_noisy = np.concatenate((X_img, noise), axis=1)
X_noisy = X_noisy[np.random.permutation(X_img.shape[0])]

X_img_reshaped = X_img.reshape(n_samples, 100, 100)
X_noisy_reshaped = X_noisy.reshape(n_samples, 100, 103)


# %%
# Step 4: Visualize original and noisy images
def plot_images(original_images, noisy_images, n_images=5):
    fig, axes = plt.subplots(2, n_images, figsize=(15, 6))

    for i in range(n_images):
        # Original images
        axes[0, i].imshow(original_images[i], cmap="gray")
        axes[0, i].set_title("Original Image")
        axes[0, i].axis("off")

        # Noisy images
        axes[1, i].imshow(noisy_images[i], cmap="gray")
        axes[1, i].set_title("Noisy Image")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()


plot_images(X_img_reshaped, X_noisy_reshaped)
# %%
print("Score avec variable de nuisance")
clf_cv.run_svm_cv(X_noisy, y)

# %%

# Q5: Score after dimensionality reduction using PCA

pca = PCA(svd_solver="randomized", random_state=12).fit(X_noisy)
var_cumu = np.cumsum(pca.explained_variance_ratio_) * 100


def num_components_exp_var(exp_var):
    var_cumu = np.cumsum(pca.explained_variance_ratio_) * 100
    # How many PCs explain 95% of the variance?
    k = np.argmax(var_cumu > exp_var)
    return k


def dim_reduction(exp_var):
    plt.figure(figsize=[10, 5])
    plt.title(
        f"Cumulative Explained Variance by {num_components_exp_var(exp_var)} components is {exp_var}%"
    )
    plt.ylabel("Cumulative Explained variance (%)")
    plt.xlabel("Principal components")
    plt.axvline(x=num_components_exp_var(exp_var), color="k", linestyle="--")
    plt.axhline(y=exp_var, color="r", linestyle="--")
    ax = plt.plot(var_cumu)
    return plt.show()


dim_reduction(95)

# %%
n_components = num_components_exp_var(95)
X_noisy_stand = scaler.fit_transform(X_noisy)
pca2 = PCA(n_components=n_components, svd_solver="randomized").fit(X_noisy_stand)
X_pca = pca2.transform(X_noisy_stand)
# %%
# %%
print("Score after dimensionality reduction")
clf_cv.run_svm_cv(X_pca, y)

#%%
plot_images(X_img_reshaped, X_pca.reshape(357, 12, 16), 5)
# %%
def plot_pca_vs_score(X, y, start_comp, end_comp, step, num_runs):
    """
    Function to plot the SVM scores as a function of the number of PCA components.

    Parameters:
    - X: Input features (before PCA)
    - y: Labels
    - start_comp: The starting number of components to test
    - end_comp: The ending number of components to test
    - step: Step size for the range of PCA components
    - num_runs: Number of times to run SVM for each component to observe score variations
    """
    scores_avg = []
    scores_std = []
    num_components_list = list(
        range(start_comp, end_comp, step)
    )  # Range of components to test

    # Standardize the data
    X_stand = scaler.fit_transform(X)

    for n_components in num_components_list:
        run_scores = []

        for run in range(num_runs):
            # Apply PCA with n_components
            pca = PCA(
                n_components=n_components, svd_solver="randomized", random_state=17
            )
            X_pca = pca.fit_transform(X_stand)

            print(f"Running SVM with {n_components} components (Run {run + 1})...")
            # Run SVM on the transformed data and capture the score
            clf_cv.run_svm_cv(X_pca, y)
            score = clf_cv.score["test_score"]
            run_scores.append(score)

        # Compute the average and standard deviation of scores across the runs
        avg_score = sum(run_scores) / len(run_scores)
        std_score = np.std(run_scores)

        scores_avg.append(avg_score)
        scores_std.append(std_score)

    # Plot the number of components vs. the SVM score (with error bars for variations)
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        num_components_list, scores_avg, yerr=scores_std, fmt="-o", color="b", capsize=5
    )
    plt.xlabel("Number of PCA Components")
    plt.ylabel("SVM Score")
    plt.title(f"SVM Score vs. Number of PCA Components (Averaged over {num_runs} runs)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return scores_avg, scores_std
# plot_pca_vs_score(X_noisy, y=y, start_comp=80, end_comp=190, step=1, num_runs=100)