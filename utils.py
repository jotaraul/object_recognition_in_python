import matplotlib.pyplot as plt
import numpy as np
import config
import pandas

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


def turn_vbles_to_categorical(data, vbles):
    n_discrete_values = config.n_discrete_values
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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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