"""
    This module is part of the set of scripts within object_recognition_in_python.
    These scripts were released by J.R. Ruiz-Sarmiento, and are publicly available at:

    https://github.com/jotaraul/object_recognition_in_python

    under the GNUv3 license. Their goal is to show some directions for the utilization
    of tools writen in Python, like pandas, seaborn or scikit-learn, for the design
    of object recognition systems.
"""

# [GENERAL CONFIGURATION]

dataset_file = '../datasets/robot_at_home.csv'
balance_dataset = False
vbles_to_work_with = [ 'PLANARITY','SCATTER','LINEARITY','MINHEIGHT','MAXHEIGHT','CENTROID_X','CENTROID_Y','CENTROID_Z',
                       'VOLUME','BIGGESTAREA','ORIENTATION','HUEMEAN','SATMEAN','VALMEAN','HUESTDV','SATSTDV','VALSTDV',
                       'HUEHIST0','HUEHIST1','HUEHIST2','HUEHIST3','HUEHIST4','VALHIST0','VALHIST1','VALHIST2',
                       'VALHIST3','VALHIST4','SATHIST0','SATHIST1','SATHIST2','SATHIST3','SATHIST4' ]
gt_vble = 'OBJECTCATEGORY'
n_object_categories = 15
show_instances_distribution = False

# [DESCRIPTIVE ANALYSIS CONFIGURATION]

describe_vbles = False
graphically_describe_vbles = False
graphical_representation = 'boxplot'
compute_chi_square_test = True
show_chi_square_test_results = False
n_discrete_values = 2

# [PRE-PROCESSING CONFIGURATION]

standarized_features = True

# [MODEL TESTING CONFIGURATION]

models_to_work_with = ['Logistic','SVM','KNeighbors','DecisionTree','RandomForest','ExtraTrees','GaussianNaiveBayes']
cross_validation_n_iterations = 10
cross_validation_n_folds = 4
show_confuion_matrix = False
confusion_matrix_show_method = 'RandomForest'
show_performance_with_sets_of_features = True

# [VISUALIZATION OPTIONS]
title_font = {'fontname': 'Arial', 'size': '21', 'color': 'black', 'weight': 'normal', 'verticalalignment': 'bottom'}
axis_font = {'fontname': 'Arial', 'size': '20'}
ticks_size = 20