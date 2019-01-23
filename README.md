# object_recognition_in_python

## Description

**New!** Now you can try the notebook using binder -> [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jotaraul/object_recognition_in_python/master?filepath=Object_recognition_in_python.ipynb)

This repository illustrates the steps needed in the design of a successful object recognition model. It mainly relies on the following python libraries:

- [pandas](https://pandas.pydata.org/) for data management.
- [seaborn](seaborn.pydata.org) for visualization purposes.
- [scikit-learn](http://scikit-learn.org/stable/) for the machine learning models.

Using that tools, and in less than 150 lines of code, it is shown how to: 

- load and manage dataset containing characterized objects, 
- analyze the dataset content and features' distribution,
- preprocess such features,
- fit different models,
- show their performance using
  - different number of cross-validation steps,
  - a confusion matrix,
  - and different features to fit the model.

For that it is provided part of the [Robot@Home](http://mapir.isa.uma.es/work/robot-at-home-dataset) dataset, facilitating the description of more than 2k object instances.

Author: J.R. Ruiz-Sarmiento [(homepage)](http://mapir.isa.uma.es/jotaraul)

<p float="left">
  <img src="https://user-images.githubusercontent.com/8874841/41841726-5724237c-7869-11e8-9710-0b9cb975ebb8.png" width="49%" />
  <img src="https://user-images.githubusercontent.com/8874841/41841729-575c6f3e-7869-11e8-823d-1c2f4e009931.png" width="49%" /> 
</p>

## Content

The tutorial is divided into three scripts, each one with a clearly defined purpose:

- **config**: a script for configuring parameters regarding the object categories to use, visualization options, recognition models, etc.
- **object_recognition**: here the magic happens. It performs the steps commented above. 
- **utils**: provides a set of functions for visualization, useful data types, etc.

## Configuration

For easying its usage, here they are described all the options provided in the configuration file:

#### [GENERAL CONFIGURATION]

- **dataset_file**: Path to the csv file containing the dataset to work with 
- **balance_dataset**: Sets if the dataset must be balanced (e.g. all the categories have the same number of instances)
- **vbles_to_work_with**: List of varibales within the dataset to consider (in capital letters). Possibilities within the Robot@Home dataset are: 'PLANARITY', 'SCATTER', 'LINEARITY', 'MINHEIGHT', 'MAXHEIGHT', 'CENTROID_X', 'CENTROID_Y', 'CENTROID_Z', 'VOLUME', 'BIGGESTAREA', 'ORIENTATION', 'HUEMEAN', 'SATMEAN', 'VALMEAN', 'HUESTDV', 'SATSTDV', 'VALSTDV', 'HUEHIST0', 'HUEHIST1', 'HUEHIST2', 'HUEHIST3', 'HUEHIST4', 'VALHIST0', 'VALHIST1', 'VALHIST2', 'VALHIST3', 'VALHIST4', 'SATHIST0', 'SATHIST1', 'SATHIST2', 'SATHIST3', 'SATHIST4'
- **gt_vble**: Variable that contains the ground truth labels in the dataset (in capital letters)
- **n_object_categories**: Sets the number of object categories to consider (the n_object_categories with more occurrences in the dataset)
- **show_instances_distribution**: Turn it on to visually see how the object categories' instances are distributed

#### [DESCRIPTIVE ANALYSIS CONFIGURATION]

- **describe_vbles**: Sets if the considered variables are described (info about count, mean, std, min, max, and quartiles is printed)
- **graphically_describe_vbles**: Enable it if you want to visually analyze the variables using... 
- **graphical_representation**: one of these graphical representations: boxplot, striplot, swarmplot, violinplot 
- **compute_chi_square_test**: Enables the computation of the chi square test, a way to check if two variables are related
- **show_chi_square_test_results**: Decides if show graphically the results of the chi square test
- **n_discrete_values**: Don't touch the parameter! Only used for chi_square computation

#### [PRE-PROCESSING CONFIGURATION]

- **standarized_features**: Standardizes variables to have a 0 mean and 1 standard deviation

#### [MODEL TESTING CONFIGURATION]

- **models_to_work_with**: Models to fit and evaluate. Some of them could need a long training time. Available models: 'Logistic', 'SVM', 'KNeighbors', 'DecisionTree', 'RandomForest', 'ExtraTrees', 'MLP', 'GaussianProcess', 'GaussianNaiveBayes', 'QDA', 'AdaBoost', 'GradientBoosting'
- **cross_validation_n_iterations**: Number of iterations of the cross validation process
- **cross_validation_n_folds**: Number of folds used: cross_validation_n_folds-1 will be used for training, the remaining one for testing
- **show_confuion_matrix**: Show a confusion matrix?
- **confusion_matrix_show_method**: If so, show the configuration matrix produced by the method...
- **show_performance_with_sets_of_features**: Computes and shows the performance of the selected models with different sets of features. For that, first all the features are used, and then iteratively the less discriminative (according to the results of the chi-square test) is removed.

#### [VISUALIZATION OPTIONS]

- **title_font**: Defines some features of the font used in the figures titles
- **axis_font**: The same for the labels in the axis
- **ticks_size** Size of the ticks labels in some figures
