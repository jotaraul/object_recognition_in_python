"""
    This module is part of the set of scripts within object_recognition_in_python.
    These scripts were released by J.R. Ruiz-Sarmiento, and are publicly available at:

    https://github.com/jotaraul/object_recognition_in_python

    under the GNUv3 license. Their goal is to show some directions for the utilization
    of tools writen in Python, like pandas, seaborn or scikit-learn, for the design
    of object recognition systems.
"""

import matplotlib.pyplot as plt
import numpy as np
from bisect import bisect
import time
import itertools

import pandas
import seaborn
from sklearn import metrics
from sklearn.model_selection import train_test_split

class statistics:
    def __init__(self):
        self.accuracy = []
        self.macro_precision = []
        self.macro_recall = []
        self.micro_precision = []
        self.f1_score = []
        self.partial_accuracy = []
        self.partial_macro_precision = []
        self.partial_f1_score = []


def turn_vbles_to_categorical(data, vbles, config):
    n_discrete_values = config["n_discrete_values"]
    quatile_ranges = np.arange(0, 1., 1.0 / n_discrete_values).tolist()
    quatile_ranges.append(1)

    for vble in vbles:
        categorical_name = vble + '_C'
        categorical_numerical_name = vble + '_C_N'
        ranges = data[vble].quantile(quatile_ranges)
        data[categorical_name] = pandas.cut(data[vble], ranges)
        data[categorical_name] = data[categorical_name].astype('category')
        quatile_ranges_c = [str(int(x * 10)) for x in quatile_ranges if x > 0]
        data[categorical_name] = data[categorical_name].cat.rename_categories(quatile_ranges_c)
        data[categorical_numerical_name] = pandas.to_numeric(data[categorical_name], errors="coerce")


def plot_instances_distribution(data, config):
    """ Plots the distribution of the instances within a given dataset according to their categories.

    Args:
        data (dataframe): Dataset loaded by pandas.

    """
    c2 = pandas.value_counts(data[config["gt_vble"]], sort=True)
    c3 = pandas.DataFrame(dict(categories=c2.index, count=c2.values)).sort_values(by=['count'], ascending=False)

    blues_pal = seaborn.color_palette("Spectral", config["n_object_categories"])
    my_pal = {}
    max = c3.values[0][1]
    for cat in c3.values:
        my_pal[cat[0]] = blues_pal[int(cat[1] * config["n_object_categories"] / max) - 1]
    ax = seaborn.barplot(x='count', y='categories', data=c3, alpha=0.8, order=c2.index, palette=my_pal)

    # ax = seaborn.barplot(x='count',y='categories', data=c3, alpha=0.8, order=c2.index)
    ax.tick_params(labelsize=config["ticks_size"])
    plt.title('Number of instances per category', **config["title_font"])
    plt.xlabel('Number of instances', **config["axis_font"])
    plt.ylabel('Object categories', **config["axis_font"])
    plt.tight_layout()
    plt.show()


def plot_variables_description(data, config):
    """ Plots, for each variable (feature) describing the objects in the dataset, a plot showing how their
    values are distributed for each object category.

    Args:
        data (dataframe): Dataset loaded by pandas.

    """
    for vble in config["vbles_to_work_with"]:
        if config["graphical_representation"] == 'boxplot':
            blues_pal = seaborn.color_palette("Spectral", 21)
            my_pal = {}
            # mean = []
            # for cat in data[config["gt_vble"]].unique():
            #     mean.append(data.loc[data[config["gt_vble"]] == cat][vble].mean())
            # l = np.arange(np.min(mean), np.max(mean), (np.max(mean) - np.min(mean)) / 20.)
            # for cat in data[config["gt_vble"]].unique():
            #     mean.append(data.loc[data[config["gt_vble"]] ==cat][vble].mean())
            l = np.arange(data[vble].min(), data[vble].max(), (data[vble].max() - data[vble].min()) / 20.)
            for cat in data[config["gt_vble"]].unique():
                print cat
                pos = (bisect(l, data.loc[data[config["gt_vble"]] == cat][vble].mean()))
                print pos
                my_pal[cat] = blues_pal[pos]
            print my_pal
            ax = seaborn.boxplot(x=vble, y=config["gt_vble"], data=data, palette=my_pal)
            ax.tick_params(labelsize='20')
            plt.xlabel(vble, **config["axis_font"])
            plt.ylabel('')
            plt.tight_layout()
            # plt.savefig('images/img-description-'+str(vble)+'.pdf', bbox_inches='tight')
        elif config["graphical_representation"] == 'striplot':
            seaborn.stripplot(x=vble, y=config["gt_vble"], data=data, jitter=True)
        elif config["graphical_representation"] == 'swarmplot':
            seaborn.swarmplot(x=vble, y=config["gt_vble"], data=data)
        elif config["graphical_representation"] == 'violinplot':
            seaborn.violinplot(x=vble, y=config["gt_vble"], data=data)
        plt.show()


def plot_chi_square_values(chi2, config):
    """ Plots the results of doing a chi-square test to all the used variables (features).

    Args:
        chi2 (list): Results of the chi-square test.

    """
    blues_pal = seaborn.color_palette("Spectral", 21)
    my_pal = {}
    minimum = np.min(chi2[0])
    maximum = np.max(chi2[0])
    ranges = np.arange(minimum, maximum, (maximum - minimum) / 20)
    for i in range(0, len(chi2[0])):
        my_pal[config["vbles_to_work_with"][i]] = blues_pal[bisect(ranges, chi2[0][i])]
    df = pandas.DataFrame(dict(chi2=chi2[0], variables=config["vbles_to_work_with"]))
    seaborn.barplot(x='chi2', y='variables', data=df, alpha=0.8, palette=my_pal)
    plt.xlabel('Chi-square results', **config["axis_font"])
    plt.show()


def plot_partial_cross_validation_results(range,
                                          metric,
                                          label):
    plt.plot(range, metric,label=label)


def plot_cross_validation(config):
    """ Configures the plot for showing the cross validation results.
    """
    ax = plt.gca()
    ax.tick_params(labelsize='16')
    if 'seaborn' in plt.style.available: plt.style.use('seaborn')
    plt.tight_layout()
    plt.legend(loc=4, prop={'size': 16})
    plt.grid(True)
    plt.xlabel('Number of iterations', **config["axis_font"])
    plt.ylabel('Mean accuracy', **config["axis_font"])
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.grid(False)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    

def plot_performance_with_sets_of_features(models,
                                           chi2,
                                           data,
					   config):
    """ Plots the performance of the given models depending on the number of features used for describing
    the features. For that, all the features are used initially to fit the models, and iteratively a feature
    is removed (according to their discriminative power reported by the chi-square test).

    Args:
        models    (list): List of models, each one represented as a dictionary.
        chi2      (list): Results of the chi-square test.
        data (dataframe): Dataset loaded by pandas.

    """
    data_X = data[config["vbles_to_work_with"]]
    data_y = data[config["gt_vble"]]
    res = [[] for i in range(0, len(models))]
    elapsed_times = []
    first = True
    print 'Testing models with different number of features: ',

    while len(chi2[0]) > 1:

        if first:
            first = False
        else:
            minimum = chi2[0].index(min(chi2[0]))
            del chi2[0][minimum]
            del chi2[1][minimum]
            data_X = data[chi2[1]]
            data_y = data[config["gt_vble"]]

        print str(len(chi2[0])) + '..',

        model_index = 0
        start = time.time()

        for model in models:

            if model['name'] in config["models_to_work_with"]:

                model['statistics'].f1_score = []
                model['statistics'].accuracy = []

                for try_i in range(0, config["cross_validation_n_iterations"]):
                    test_size = 1. / config["cross_validation_n_folds"]
                    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=test_size)

                    model['clf'].fit(X_train, y_train)
                    y_pred = model['clf'].predict(X_test)
                    model['statistics'].f1_score.append(metrics.f1_score(y_test, y_pred, average='micro'))
                    model['statistics'].accuracy.append(metrics.accuracy_score(y_test, y_pred))

                #print len(model['statistics'].f1_score)

                res[model_index].append(np.mean(model['statistics'].f1_score))

            model_index += 1

        end = time.time()
        elapsed_times.append(end-start)

    x_axis = list(np.arange(len(config["vbles_to_work_with"]), 0, -1))

    seaborn.set_palette(seaborn.color_palette("husl", len(config["models_to_work_with"])))
    # print '\nElapsed times: ' + str(elapsed_times)
    # print 'Results:'

    for i in range(len(res)):
        if len(res[i]):
    #        print models[i]['name'] + ': ' + str(res[i])
            plt.plot(x_axis, res[i], label=models[i]['name'])

    if 'seaborn' in plt.style.available: plt.style.use('seaborn')
    ax = plt.gca()
    ax.tick_params(labelsize='16')
    plt.xlabel('Number of features', **config["axis_font"])
    plt.ylabel('Mean accuracy', **config["axis_font"])
    plt.legend(loc=4,prop={'size': 16})
    plt.grid(True)
    plt.tight_layout()
    plt.show()