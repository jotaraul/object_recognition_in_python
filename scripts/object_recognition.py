# -*- coding: utf-8 -*-

"""
    This module is part of the set of scripts within object_recognition_in_python.
    These scripts were released by J.R. Ruiz-Sarmiento, and are publicly available at:

    https://github.com/jotaraul/object_recognition_in_python

    under the GNUv3 license. Their goal is to show some directions for the utilization
    of tools writen in Python, like pandas, seaborn or scikit-learn, for the design
    of object recognition systems.
"""

import matplotlib.pyplot as plt
# General imports
import numpy as np
# For managing and ploting data
import pandas
import scipy
import seaborn
import time
from bisect import bisect
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# scikit-learn imports
from sklearn import svm
from sklearn import tree
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Tutorial local imports
import utils
import config

# ---------------------------------------------------------------------------------------------------------------------
# First step: data managament and analysis
# ---------------------------------------------------------------------------------------------------------------------

# Let's go! Load the dataset
data = pandas.read_csv(config.dataset_file, low_memory=False)

# print some information about the dataset
print "[Working the the '" + config.dataset_file.split('/')[-1] + "' dataset!]"
print "Info: "
print "Number of observations within the dataset: " + str(len(data))  # number of observations (rows)
print "Number of variables                      : " + str(len(data.columns))  # number of variables (columns)

# upper-case all DataFrame column names
data.columns = map(str.upper, data.columns)

for vble in config.vbles_to_work_with:
    # setting variables you will be working with to numeric
    data[vble] = pandas.to_numeric(data[vble], errors="coerce")

# And the ground truth one to categorical
data[config.gt_vble] = data[config.gt_vble].astype('category')

# Get entries corresponding with the selected object categories (config.n_object_categories most appearing)
sub_data = pandas.value_counts(data[config.gt_vble], sort=True)
print "Number of observations with values       : " + str(np.sum(sub_data))
print 'Number of different categories           : ' + str(sub_data.size)
print "\nNumber of objects per category in the dataset:"
print sub_data
data = data[data['OBJECTCATEGORY'].isin(sub_data.index[0:config.n_object_categories])]
data['OBJECTCATEGORY'] = data['OBJECTCATEGORY'].cat.remove_unused_categories()
data[config.gt_vble] = data[config.gt_vble].replace({'upper_cabinet': 'ucabinet'}) # only for Robot@Home dataset

if config.show_instances_distribution:
    # Visually show the most appearing categories
    utils.plot_instances_distribution(data)

if config.balance_dataset:
    g = data.groupby('OBJECTCATEGORY')
    data = g.apply(lambda x: x.sample(g.size().min())).reset_index(drop=True)

    c2 = pandas.value_counts(data[config.gt_vble], sort=True)
    print "Number of observations with values: " + str(np.sum(c2))
    print c2
    data = data[data['OBJECTCATEGORY'].isin(c2.index[0:config.n_object_categories])]
    data['OBJECTCATEGORY'] = data['OBJECTCATEGORY'].cat.remove_unused_categories()

if config.describe_vbles:
    # Describe predictors
    for vble in config.vbles_to_work_with:
        print data[vble].describe()

if config.graphically_describe_vbles:
    # Describe predictors graphically
    utils.plot_variables_description()

if config.compute_chi_square_test:
    # Turn predictors into categorical vbles for chi-square test
    utils.turn_vbles_to_categorical(data, config.vbles_to_work_with)
    chi2 = [[], []]

    for vble in config.vbles_to_work_with:
        ct1 = pandas.crosstab(data[config.gt_vble], data[vble + '_C'])
        print str(vble) + '_C: chi-square value, p value, expected counts'
        cs1 = scipy.stats.chi2_contingency(ct1)
        chi2[0].append(cs1[0])
        chi2[1].append(vble)
        print cs1

    if config.show_chi_square_test_results:
        utils.plot_chi_square_results(chi2)

# ---------------------------------------------------------------------------------------------------------------------
# Second step: preprocessing
# ---------------------------------------------------------------------------------------------------------------------

if config.standarized_features:
    # Standardize variables to have mean=0 and sd=1
    for vble in config.vbles_to_work_with:
        data[vble] = preprocessing.scale(data[vble].astype('float64'))

# ---------------------------------------------------------------------------------------------------------------------
# Third step: model fitting and evaluation
# ---------------------------------------------------------------------------------------------------------------------

# Prepare predictor and ground truth variables
data_X = data[config.vbles_to_work_with]
data_y = data[config.gt_vble]

# Compute the performance of each model
models = [{'name': 'Logistic', 'clf': linear_model.LogisticRegression(), 'statistics': utils.statistics()},
          {'name': 'DecisionTree', 'clf': tree.DecisionTreeClassifier(), 'statistics': utils.statistics()},
          {'name': 'RandomForest', 'clf': RandomForestClassifier(), 'statistics': utils.statistics()},
          {'name': 'ExtraTrees', 'clf': ExtraTreesClassifier(), 'statistics': utils.statistics()},
          {'name': 'SVM', 'clf': svm.SVC(kernel='linear'), 'statistics': utils.statistics()},
          {'name': 'KNeighbors', 'clf': KNeighborsClassifier(), 'statistics': utils.statistics()},
          {'name': 'MLP', 'clf': MLPClassifier(alpha=1), 'statistics': utils.statistics()},
          {'name': 'GaussianProcess', 'clf': GaussianProcessClassifier(1.0 * RBF(1.0)), 'statistics': utils.statistics()},
          {'name': 'GaussianNaiveBayes', 'clf': GaussianNB(), 'statistics': utils.statistics()},
          {'name': 'QDA', 'clf': QuadraticDiscriminantAnalysis(), 'statistics': utils.statistics()},
          {'name': 'AdaBoost', 'clf': AdaBoostClassifier(), 'statistics': utils.statistics()},
          {'name': 'GradientBoosting', 'clf': GradientBoostingClassifier(), 'statistics': utils.statistics()}
          ]

if config.show_confuion_matrix:
    accumulated_cnf_matrix = np.zeros(shape=(config.n_object_categories, config.n_object_categories), dtype=int)

print 'Obtaining metrics for..',

for model in models:

    if model['name'] in config.models_to_work_with:

        seaborn.set_palette(seaborn.color_palette("husl", len(config.models_to_work_with)))

        print model['name'] + '..',

        for try_i in range(0, config.cross_validation_n_iterations):

            test_size = 1. / config.cross_validation_n_folds
            X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=test_size)
            model['clf'].fit(X_train, y_train)
            y_pred = model['clf'].predict(X_test)

            model['statistics'].accuracy.append(metrics.accuracy_score(y_test, y_pred))
            model['statistics'].macro_precision.append(metrics.precision_score(y_test, y_pred, average='macro'))
            model['statistics'].macro_recall.append(metrics.recall_score(y_test, y_pred, average='macro'))
            model['statistics'].f1_score.append(metrics.f1_score(y_test, y_pred, average='micro'))

            model['statistics'].partial_accuracy.append(np.mean(model['statistics'].accuracy))
            model['statistics'].partial_macro_precision.append(np.mean(model['statistics'].macro_precision))
            model['statistics'].partial_f1_score.append(np.mean(model['statistics'].f1_score))

            if config.show_confuion_matrix and model['name'] == config.confusion_matrix_show_method:
                cnf_matrix = confusion_matrix(y_test, y_pred)
                accumulated_cnf_matrix = np.add(accumulated_cnf_matrix, cnf_matrix)

        utils.plot_partial_cross_validation_results(range=range(1, config.cross_validation_n_iterations + 1),
                                                    metric=model['statistics'].partial_f1_score,
                                                    label=model['name'])

utils.plot_cross_validation()

if config.show_confuion_matrix:
    # Show confusion matrix
    utils.plot_confusion_matrix(accumulated_cnf_matrix, classes=sorted(c2.index[0:config.n_object_categories]),
                                normalize=False,
                                title='Normalized confusion matrix')
    utils.plot_confusion_matrix(accumulated_cnf_matrix, classes=sorted(c2.index[0:config.n_object_categories]),
                                normalize=True,
                                title='Normalized confusion matrix')

if config.show_performance_with_sets_of_features:
    # Show the models performance using a different sets of features
    utils.plot_performance_with_sets_of_features(models, chi2, data)




    # elif model == "Voting":
    #     #clf1 = LogisticRegression(random_state=1)
    #     clf1 = GradientBoostingClassifier(random_state=1)
    #     clf2 = RandomForestClassifier(random_state=1)
    #     clf3 = GaussianNB()
    #     clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')



    # if model == 'DecisionTree':
    #    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=.33, random_state=123)
    #    clf = tree.DecisionTreeClassifier()
    #    clf.fit(X_train,y_train)
    #    print clf.feature_importances_
    #    y_pred = clf.predict(X_test)
    #    y_true = y_test

    #    print 'Accuracy: ' + str(metrics.accuracy_score(y_true, y_pred))
    #    print metrics.classification_report(y_true, y_pred, target_names=sorted(c2.index[0:config.n_object_categories]))

    # from io import StringIO
    # from io import StringIO
    # out = StringIO()
    # with open("classifier.txt", "w") as f:
    #    f = tree.export_graphviz(clf, out_file=f)



