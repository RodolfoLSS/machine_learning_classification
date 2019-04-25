import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from ml_bc_pipeline.utils import CustomScaler
from ml_bc_pipeline.utils import BalanceDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, average_precision_score, precision_recall_curve
from keras import layers
from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier

def grid_search_MLP(training, param_grid, seed, cv=5):
    """ Multi-layer Perceptron classifier hyperparameter estimation using grid search with cross-validation.

    In this function, the MLP classifier is optimized by CV, implemented through GridSearchCV function from
    sklearn. Semantically, i.e., not technically, this is performed in the following way:
     1) several models are created with different hyper-parameters (according to param_grid);
     2) their performance is assessed by means of k-fold cross-validation (k=cv):
        2) 1) for cv times, the model is trained using k-1 folds of the training data;
        2) 2) each time, the resulting model is validated on the held out (kth) part of the data;
        2) 3) the final performance is computed as the average along cv iterations.


    From theory it is known that input standardization allows an ANN perform better. For this reason, this
    function automatically embeds input standardization within hyperparameter estimation procedure. This is
    done by arranging sklearn.preprocessing.StandardScaler and sklearn.neural_network.MLPClassifier into the
    same "pipeline". The tool which allows to do so is called sklearn.pipeline.Pipeline. More specifically,
    the preprocessing module further provides a utility class StandardScaler that implements the Transformer
    API to compute the mean and standard deviation on a training set so as to be able to later reapply the
    same transformation on the testing set.
    """

    dummies = list(training.select_dtypes(include=["category", "object"]).columns)
    if not dummies:
        pipeline = Pipeline([("std_scaler", StandardScaler()), ("mlpc", MLPClassifier(random_state=seed))])
    else:
        filt = ~ training.loc[:, training.columns != "Response"].columns.isin(dummies)
        continuous_idx = np.arange(0, len(filt))[filt]
        not_filt = [not i for i in filt]
        dummies_idx = np.arange(0, len(filt))[not_filt]
        pipeline = Pipeline(
            [("std_scaler", CustomScaler(continuous_idx, dummies_idx)), ("mlpc", MLPClassifier(random_state=seed))])

    clf_gscv = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring=make_scorer(average_precision_score))
    clf_gscv.fit(training.loc[:, training.columns != "Response"].values, training["Response"].values)

    return clf_gscv


def assess_generalization_auprc(estimator, unseen):
    y_score = estimator.predict_proba(unseen.loc[:, unseen.columns != "Response"].values)[:, 1]
    precision, recall, thresholds = precision_recall_curve(unseen["Response"], y_score)
    auc = average_precision_score(unseen["Response"], y_score, average="weighted")

    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, marker='.', label=" (AUPR (unseen) {:.2f}".format(auc) + ")")
    plt.plot([0, 1], [0.5, 0.5], 'k--')
    plt.xlabel('Recall (unseen)')
    plt.ylabel('Precision (unseen)')
    plt.title('PR curve on unseen data')
    plt.legend(loc='best', title="Models")

    return auc

def create_model_chis():
    # create model
    model = Sequential()
    model.add(layers.Dense(100,activation='relu', input_dim=47))
    model.add(layers.Dropout(rate = 0.2))
    model.add(layers.Dense(100,activation='relu'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_model_dta():
    # create model
    model = Sequential()
    model.add(layers.Dense(200,activation='relu', input_dim=9))
    model.add(layers.Dropout(rate = 0.2))
    model.add(layers.Dense(200,activation='relu'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def grid_search_NN(training, param_grid, type, cv=5):
    # fix random seed for reproducibility
    if type == "dta":
        model = KerasClassifier(build_fn=create_model_dta)
    else:
        model = KerasClassifier(build_fn=create_model_chis)

    dummies = list(training.select_dtypes(include=["category", "object"]).columns)
    if not dummies:
        pipeline = Pipeline([("std_scaler", StandardScaler()), ("nn", model)])
    else:
        filt = ~ training.loc[:, training.columns != "Response"].columns.isin(dummies)
        continuous_idx = np.arange(0, len(filt))[filt]
        not_filt = [not i for i in filt]
        dummies_idx = np.arange(0, len(filt))[not_filt]
        pipeline = Pipeline(
            [("std_scaler", CustomScaler(continuous_idx, dummies_idx)),("nn", model)])

    # define the grid search parameters
    training = BalanceDataset(training)
    nn_gscv = GridSearchCV(pipeline, cv=cv, param_grid=param_grid,
                           n_jobs=-1, scoring=make_scorer(average_precision_score))
    nn_gscv.fit(training.loc[:, training.columns != "Response"].values, training["Response"].values)
    # summarize results
    return nn_gscv

def grid_search_SVM(training, param_grid, seed, cv=5):

    dummies = list(training.select_dtypes(include=["category", "object"]).columns)
    if not dummies:
        pipeline = Pipeline([("std_scaler", StandardScaler()), ("svm", SVC())])
    else:
        filt = ~ training.loc[:, training.columns != "Response"].columns.isin(dummies)
        continuous_idx = np.arange(0, len(filt))[filt]
        not_filt = [not i for i in filt]
        dummies_idx = np.arange(0, len(filt))[not_filt]
        pipeline = Pipeline(
            [("std_scaler", CustomScaler(continuous_idx, dummies_idx)), ("svm", SVC())])

    svm_gscv = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring=make_scorer(average_precision_score))
    svm_gscv.fit(training.loc[:, training.columns != "Response"].values, training["Response"].values)

    return svm_gscv

def grid_search_KNN(training, param_grid, seed, cv=5):

    dummies = list(training.select_dtypes(include=["category", "object"]).columns)
    if not dummies:
        pipeline = Pipeline([("std_scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
    else:
        filt = ~ training.loc[:, training.columns != "Response"].columns.isin(dummies)
        continuous_idx = np.arange(0, len(filt))[filt]
        not_filt = [not i for i in filt]
        dummies_idx = np.arange(0, len(filt))[not_filt]
        pipeline = Pipeline(
            [("std_scaler", CustomScaler(continuous_idx, dummies_idx)), ("knn", KNeighborsClassifier())])

    knn_gscv = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring=make_scorer(average_precision_score))
    knn_gscv.fit(training.loc[:, training.columns != "Response"].values, training["Response"].values)

    return knn_gscv

def grid_search_DT(training, param_grid, seed, cv=5):

    dummies = list(training.select_dtypes(include=["category", "object"]).columns)
    if not dummies:
        pipeline = Pipeline([("dt", DecisionTreeClassifier(random_state=seed))])
    else:
        filt = ~ training.loc[:, training.columns != "Response"].columns.isin(dummies)
        continuous_idx = np.arange(0, len(filt))[filt]
        not_filt = [not i for i in filt]
        dummies_idx = np.arange(0, len(filt))[not_filt]
        pipeline = Pipeline(
            [("dt", DecisionTreeClassifier(random_state=seed))])

    #training = BalanceDataset(training)
    dt_gscv = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring=make_scorer(average_precision_score))
    dt_gscv.fit(training.loc[:, training.columns != "Response"].values, training["Response"].values)

    return dt_gscv

def grid_search_RF(training, param_grid, seed, cv=5):

    dummies = list(training.select_dtypes(include=["category", "object"]).columns)
    if not dummies:
        pipeline = Pipeline([("rf", RandomForestClassifier())])
    else:
        filt = ~ training.loc[:, training.columns != "Response"].columns.isin(dummies)
        continuous_idx = np.arange(0, len(filt))[filt]
        not_filt = [not i for i in filt]
        dummies_idx = np.arange(0, len(filt))[not_filt]
        pipeline = Pipeline(
            [("rf", RandomForestClassifier())])

    rf_gscv = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring=make_scorer(average_precision_score))
    rf_gscv.fit(training.loc[:, training.columns != "Response"].values, training["Response"].values)

    return rf_gscv

def grid_search_NB(training, param_grid, seed, cv=5):

    dummies = list(training.select_dtypes(include=["category", "object"]).columns)
    if not dummies:
        pipeline = Pipeline([("std_scaler", StandardScaler()), ("nb", GaussianNB())])
    else:
        filt = ~ training.loc[:, training.columns != "Response"].columns.isin(dummies)
        continuous_idx = np.arange(0, len(filt))[filt]
        not_filt = [not i for i in filt]
        dummies_idx = np.arange(0, len(filt))[not_filt]
        pipeline = Pipeline(
            [("std_scaler", CustomScaler(continuous_idx, dummies_idx)), ("nb", GaussianNB())])

    training = BalanceDataset(training)
    nb_gscv = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring=make_scorer(average_precision_score))
    nb_gscv.fit(training.loc[:, training.columns != "Response"].values, training["Response"].values)

    return nb_gscv

def grid_search_LR(training, param_grid, seed, cv=5):

    dummies = list(training.select_dtypes(include=["category", "object"]).columns)
    if not dummies:
        pipeline = Pipeline([("std_scaler", StandardScaler()), ("nb", LogisticRegression())])
    else:
        filt = ~ training.loc[:, training.columns != "Response"].columns.isin(dummies)
        continuous_idx = np.arange(0, len(filt))[filt]
        not_filt = [not i for i in filt]
        dummies_idx = np.arange(0, len(filt))[not_filt]
        pipeline = Pipeline(
            [("std_scaler", CustomScaler(continuous_idx, dummies_idx)), ("nb", LogisticRegression())])

    #training = BalanceDataset(training)
    lr_gscv = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring=make_scorer(average_precision_score))
    lr_gscv.fit(training.loc[:, training.columns != "Response"].values, training["Response"].values)

    return lr_gscv

def Voting(training, clfs, seed, cv=5):

    clfs["voting"] = VotingClassifier(estimators=list(clfs.items()), voting='soft',
                                      scoring=make_scorer(average_precision_score))
    print(clfs)

    #for name, clf in clfs.items():
        #clf.fit(training.loc[:, training.columns != "Response"].values, training["Response"].values)


    #return clf

def grid_search_BAG(training, param_grid, voters, seed, cv=5):

    dummies = list(training.select_dtypes(include=["category", "object"]).columns)
    if not dummies:
        pipeline = Pipeline([("bag", BaggingClassifier())])
    else:
        filt = ~ training.loc[:, training.columns != "Response"].columns.isin(dummies)
        continuous_idx = np.arange(0, len(filt))[filt]
        not_filt = [not i for i in filt]
        dummies_idx = np.arange(0, len(filt))[not_filt]
        pipeline = Pipeline(
            [("bag", VotingClassifier(estimators=voters, voting='soft'))])

    bag_gscv = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring=make_scorer(average_precision_score))
    bag_gscv.fit(training.loc[:, training.columns != "Response"].values, training["Response"].values)

    return bag_gscv


