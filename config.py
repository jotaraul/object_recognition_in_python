# -*- coding: utf-8 -*-

# Configuration file

# [GENERAL CONFIGURATION]

# Path to the csv file containing the dataset to work with
dataset_file = 'datasets/robot_at_home.csv'
# Sets if the dataset must be balanced (e.g. all the categories have the same number of instances)
balance_dataset = False
# List of varibales within the dataset to consider (in capital letters)
# Possibilities within the Robot@Home dataset are:
vbles_to_work_with = [ 'PLANARITY','SCATTER','LINEARITY','MINHEIGHT','MAXHEIGHT',
 'CENTROID_X','CENTROID_Y','CENTROID_Z','VOLUME','BIGGESTAREA','ORIENTATION',
 'HUEMEAN','SATMEAN','VALMEAN','HUESTDV','SATSTDV','VALSTDV',
 'HUEHIST0','HUEHIST1','HUEHIST2','HUEHIST3','HUEHIST4',
 'VALHIST0','VALHIST1','VALHIST2','VALHIST3','VALHIST4',
 'SATHIST0','SATHIST1','SATHIST2','SATHIST3','SATHIST4']
#vbles_to_work_with = ['PLANARITY','SCATTER','LINEARITY','CENTROID_X','HUEMEAN','MINHEIGHT','MAXHEIGHT','VOLUME','BIGGESTAREA','ORIENTATION']
#vbles_to_work_with = ['HUEMEAN','MINHEIGHT','MAXHEIGHT','BIGGESTAREA','ORIENTATION','VOLUME','PLANARITY']
# Variable that contains the ground truth labels (in capital letters)
gt_vble = 'OBJECTCATEGORY'
# Sets the number of object categories to consider (the n_object_categories with more occurrences in the dataset)
n_object_categories = 15
# Turn it on to visually see how the object categories' instances are distributed
show_instances_distribution = False

# [DESCRIPTIVE ANALYSIS CONFIGURATION]

# Sets if the considered variables are described (info about count, mean, std, min, max, and quartiles is printed)
describe_vbles = False
# Enable it if you want to visually analyze the variables using...
graphically_describe_vbles = False
# one of these graphical representations: boxplot, striplot, swarmplot, violinplot
graphical_representation = 'boxplot'
# Enables the computation of the chi square test, a way to check if two variables are related
compute_chi_square_test = True
# Decides if show graphically the results of the chi square test
show_chi_square_test_results = False
# Don't touch the parameter! Only used for chi_square computation
n_discrete_values = 2

# [PRE-PROCESSING CONFIGURATION]

#standardize variables to have a 0 mean and 1 standard deviation
standarized_features = True

# [MODEL TESTING CONFIGURATION]
# Available models: ['Logistic','SVM','KNeighbors','DecisionTree','RandomForest','ExtraTrees','MLP','GaussianProcess',
# 'GaussianNaiveBayes','QDA','AdaBoost','GradientBoosting']
models_to_work_with = ['Logistic','SVM','KNeighbors','DecisionTree','RandomForest','ExtraTrees','GaussianNaiveBayes']
# Number of iterations of the cross validation process
cross_validation_n_iterations = 500
# Number of folds used: cross_validation_n_folds-1 will be used for training, the remaining one for testing
cross_validation_n_folds = 4
# Show a confusion matrix?
show_confuion_matrix = False
# If so, show the configuration matrix produced by the method...
confusion_matrix_show_method = 'RandomForest'
# Compute and show the performance of the selected models with different sets of features
show_performance_with_sets_of_features = True

# [VISUALIZATION OPTIONS]
title_font = {'fontname': 'Arial', 'size': '21', 'color': 'black', 'weight': 'normal', 'verticalalignment': 'bottom'}
axis_font = {'fontname': 'Arial', 'size': '20'}
ticks_size = 20