#!/usr/bin/env/python3.5
import mnist_loader
import network2 as net
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from timeit import default_timer

# read the data
TRAINING_DATA, VALIDATION_DATA, TEST_DATA = mnist_loader.load_data_wrapper()
TRAINING_DATA, VALIDATION_DATA, TEST_DATA = list(TRAINING_DATA), list(VALIDATION_DATA), list(TEST_DATA)

# size of data sets
TRAINING_SIZE, TEST_SIZE = len(TRAINING_DATA), len(TEST_DATA)

# smaller subset for faster tuning
N_VAL_SAMPLES = 1000
SMALL_VALIDATION_DATA = VALIDATION_DATA[:N_VAL_SAMPLES]
SMALL_TRAINING_DATA = TRAINING_DATA[:5000]

# initialize constants
LAYERS = [784, 30, 10]

# training parameters
TRAINING_EPOCHS, TEST_EPOCHS = 30, 100


# initial grid search values
INITIAL_LMBDA = 0  # used for LR search
INITIAL_LEARNING_RATES = np.logspace(-3, 3, 7)
INITIAL_LAMBDAS = np.logspace(-3, 2, 6)


# UDFs
def get_training_parameters():
    """passes the function args without storing the dict & large data in memory"""
    return {'training_data': SMALL_TRAINING_DATA,
            'epochs': TRAINING_EPOCHS,
            'mini_batch_size': 10,
            'evaluation_data': SMALL_VALIDATION_DATA,
            'early_stopping_n': 10,
            'monitor_evaluation_cost': True,
            'monitor_evaluation_accuracy': True,
            'monitor_training_cost': True,
            'monitor_training_accuracy': True}


def get_test_parameters():
    """passes the function args without storing the dict & large data in memory"""
    return {
        'training_data': TRAINING_DATA,
        'epochs': TEST_EPOCHS,
        'mini_batch_size': 10,
        'evaluation_data': TEST_DATA,
        'early_stopping_n': 33,
        'monitor_evaluation_cost': True,
        'monitor_evaluation_accuracy': True,
        'monitor_training_cost': True,
        'monitor_training_accuracy': True
    }


def plot_search_results(loop_results_dict, parameter_value_list, sample_size=N_VAL_SAMPLES, epochs=TRAINING_EPOCHS):
    """
    Takes as inputs the dict that is created from the parameter loop
    and the list of values that was looped through. Use the sample_size parameter to
    compute the accuracy
    It creates a long-form df used to output a seaborn multi-line chart
    that compares the variable's effect on the y-value.
    """
    # create df for plot

    df = pd.DataFrame.from_dict(loop_results_dict)
    df['epoch'] = np.arange(epochs)
    df_long = pd.melt(df, id_vars='epoch', value_vars=parameter_value_list)
    df_long['variable'] = df_long.variable.astype('float64')
    df_long['value'] = df_long.value / sample_size

    # create plot
    plt.gcf().clear()
    colors = sns.color_palette('muted', n_colors=len(parameter_value_list))
    sns.lineplot(x='epoch', y='value', hue='variable', palette=colors, data=df_long)


def plot_test_results(results_dict, train_size=TRAINING_SIZE, test_size=TEST_SIZE):
    """
    Takes as inputs a dict that contains lists of the train and test accuracy numbers
    Computes the accuracy & creates a long-form df used to output a seaborn multi-line chart,
    Comparing the training and test accuracies
    """
    parameter_value_list=['train', 'test']
    df = pd.DataFrame.from_dict(results_dict)
    test_epochs = len(df)

    df['epoch'], df['train'], df['test'] = np.arange(test_epochs), df['train'] / train_size, df['test'] / test_size
    df_long = pd.melt(df, id_vars='epoch', value_vars=parameter_value_list)

    # create plot
    plt.gcf().clear()
    colors = sns.color_palette('muted', n_colors=len(parameter_value_list))
    sns.lineplot(x='epoch', y='value', hue='variable', palette=colors, data=df_long)


def append_nones(a_list, epochs=TRAINING_EPOCHS):
    """Takes a list and appends the number of Nones needed"""
    if len(a_list) == epochs:
        return a_list
    else:
        n_missing_items = epochs - len(a_list)
        n_nones = [None] * n_missing_items
        return a_list + n_nones


def execute_parameter_search(search_parameter, search_values, cost=net.QuadraticCost, the_best_lr=None):
    """
    Tnis is the exercise's workhorse. It conducts the parameter grid searches.

    """
    assert search_parameter in ['lr', 'lambda'], 'Please select either "lambda" or "lr" for the search_parameter'
    start_time = default_timer()
    results_dict = {}

    for search_value in search_values:
        print(search_value)
        if search_parameter == 'lr':
            search_parameter_dict = {'eta': eval('search_value'), 'lmbda': INITIAL_LMBDA}
        else:
            assert isinstance(the_best_lr, float), "If searching for the L2, please provide the learning rate."
            search_parameter_dict = {'eta': the_best_lr, 'lmbda': eval('search_value')}
        the_net = net.Network(LAYERS, cost=cost)  # initialize network
        validation_results = the_net.SGD(**search_parameter_dict, **get_training_parameters()) # run the gradient descent
        results_dict[search_value] = append_nones(validation_results[1])  # store validation accuracy
        print('--LOOP END-- minutes so far:', round((default_timer() - start_time) / 60, 1))
    print('The %s %s search has completed for these values: ' % (cost.__name__, search_parameter), search_values)
    return results_dict

#### Quadratic Cost

# QC 1ST Learning Rate Set
quadratic_LR_val_accuracy_1 = execute_parameter_search('lr', INITIAL_LEARNING_RATES)
# plot the validation accuracy curves
plot_search_results(quadratic_LR_val_accuracy_1, INITIAL_LEARNING_RATES)
# select the lr
the_best_quadratic_LR_so_far = INITIAL_LEARNING_RATES[2]  # .1 is best


# QC 2ND Learning Rate Set
# initialize search parameters
quadratic_LRs_2 = np.linspace(the_best_quadratic_LR_so_far / 2, the_best_quadratic_LR_so_far * 4, 5)
# run the SGD loop
quadratic_LR_val_accuracy_2 = execute_parameter_search('lr', quadratic_LRs_2)
# plot the validation accuracy curves
plot_search_results(quadratic_LR_val_accuracy_2, quadratic_LRs_2)
# set the QC lr
the_quadratic_lr = quadratic_LRs_2[4] # .4 is best


# QC L2 Regularization
# QC 1ST lambdas set
quadratic_lmbda_val_accuracy_1 = execute_parameter_search('lambda', INITIAL_LAMBDAS, the_best_lr=the_quadratic_lr)
# plot the validation accuracy curves
plot_search_results(quadratic_lmbda_val_accuracy_1, INITIAL_LAMBDAS)
# select the best one
the_best_quadratic_lmbda_so_far = INITIAL_LAMBDAS[1]  # .01 is best

# QC 2ND lambdas set
# initialize search parameters
quadratic_lmbdas_2 = np.linspace(the_best_quadratic_lmbda_so_far / 2, the_best_quadratic_lmbda_so_far * 20, 5)
# run the SGD loop
quadratic_lmbda_val_accuracy_2 = execute_parameter_search('lambda', quadratic_lmbdas_2, the_best_lr=the_quadratic_lr)
# plot the validation accuracy curves
plot_search_results(quadratic_lmbda_val_accuracy_2, quadratic_lmbdas_2)
# set the L2 value
the_quadratic_lmbda = quadratic_lmbdas_2[0]  # 0.005 is best



#### Cross Entropy Cost

# CE 1ST Learning Rate Set
cross_ent_LR_val_accuracy_1 = execute_parameter_search('lr', INITIAL_LEARNING_RATES, cost=net.CrossEntropyCost)
# plot the validation accuracy curves
plot_search_results(cross_ent_LR_val_accuracy_1, INITIAL_LEARNING_RATES)
# select the lr
the_best_cross_ent_LR_so_far = INITIAL_LEARNING_RATES[1]  # .01 is best


# CE 2ND Learning Rate Set
# initialize search parameters
cross_ent_LRs_2 = np.linspace(the_best_cross_ent_LR_so_far / 1.25, the_best_cross_ent_LR_so_far * 20, 5)
# run the SGD loop
cross_ent_LR_val_accuracy_2 = execute_parameter_search('lr', cross_ent_LRs_2, cost=net.CrossEntropyCost)
# plot the validation accuracy curves
plot_search_results(cross_ent_LR_val_accuracy_2, cross_ent_LRs_2)
# set the CE lr
the_cross_ent_lr = cross_ent_LRs_2[1] # 0.056 is best


# CE L2 Regularization
# CE 1ST lambdas set
cross_ent_lmbda_val_accuracy_1 = execute_parameter_search('lambda', INITIAL_LAMBDAS[:-2], the_best_lr=the_cross_ent_lr,
                                                          cost=net.CrossEntropyCost)
# plot the validation accuracy curves
plot_search_results(cross_ent_lmbda_val_accuracy_1, INITIAL_LAMBDAS)
# select the best one
the_best_cross_ent_lmbda_so_far = INITIAL_LAMBDAS[1]  # .01 is best

# CE 2ND lambdas set
# initialize search parameters
cross_ent_lmbdas_2 = np.linspace(the_best_cross_ent_lmbda_so_far / 30, the_best_cross_ent_lmbda_so_far * 30, 5)
# run the SGD loop
cross_ent_lmbda_val_accuracy_2 = execute_parameter_search('lambda', cross_ent_lmbdas_2, the_best_lr=the_cross_ent_lr,
                                                          cost=net.CrossEntropyCost)
# plot the validation accuracy curves
plot_search_results(cross_ent_lmbda_val_accuracy_2, cross_ent_lmbdas_2)
# set the L2 value
the_cross_ent_lmbda = cross_ent_lmbdas_2[1]  # 0.07525 is best



#### TEST DATA
# Quadratic-Sigmoid test
test_start_time = default_timer()
net_quadratic = net.Network(LAYERS, cost=net.QuadraticCost)
quadratic_test_results = net_quadratic.SGD(eta=the_quadratic_lr, lmbda=the_quadratic_lmbda, **get_test_parameters())
print('\n', round((default_timer() - test_start_time) / 60, 1))
# store the test results
quadratic_test_results_dict = {
    'train': quadratic_test_results[3],
    'test': quadratic_test_results[1] 
}
# plot the test results
plot_test_results(quadratic_test_results_dict)


# Cross Entropy-Sigmoid test
test_start_time = default_timer()
net_cross_ent= net.Network(LAYERS, cost=net.CrossEntropyCost)
cross_ent_test_results = net_cross_ent.SGD(eta=the_cross_ent_lr, lmbda=the_cross_ent_lmbda, **get_test_parameters())
print('\n', round((default_timer() - test_start_time) / 60, 1))
# store the test results
cross_ent_test_results_dict = {
    'train': CE_test_results[3],
    'test': CE_test_results[1] 
}
# plot the test results
plot_test_results(cross_ent_test_results_dict)
