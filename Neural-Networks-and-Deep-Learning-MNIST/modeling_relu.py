import my_network2 as net

def execute_parameter_search(search_parameter, search_values, cost=net.QuadraticCost, the_best_lr=None):
    """"""
    assert search_parameter in ['lr', 'lambda'], 'Please select either "lambda" or "lr" for the search_parameter'
    start_time = default_timer()
    results_dict = {}

    # select the activation function for Network
    if cost.__name__ == "QuadraticCostReLU":
        activation_fn = net.relu
    else:
        activation_fn = net.sigmoid

    for search_value in search_values:
        print(search_value)
        if search_parameter == 'lr':
            search_parameter_dict = {'eta': eval('search_value'), 'lmbda': INITIAL_LMBDA}
        else:
            assert isinstance(the_best_lr, float), "If searching for the L2, please provide the learning rate."
            search_parameter_dict = {'eta': the_best_lr, 'lmbda': eval('search_value')}
        the_net = net.Network(LAYERS, cost=cost, activation_fn=activation_fn)  # initialize network
        validation_results = the_net.SGD(**search_parameter_dict, **get_training_parameters()) # run the gradient descent
        results_dict[search_value] = append_nones(validation_results[1])  # store validation accuracy
        print('--LOOP END-- minutes so far:', round((default_timer() - start_time) / 60, 1))
    print('The %s %s search has completed for these values: ' % (cost.__name__, search_parameter), search_values)
    return results_dict


#### Cross Entropy Cost w/ ReLU

# CE 1ST Learning Rate Set
cross_ent_relu_LR_val_accuracy_1 = execute_parameter_search('lr', INITIAL_LEARNING_RATES, cost=net.CrossEntropyCostReLU)
# plot the validation accuracy curves
plot_search_results(cross_ent_relu_LR_val_accuracy_1, INITIAL_LEARNING_RATES)
# select the lr
the_best_cross_ent_relu_LR_so_far = INITIAL_LEARNING_RATES[1]  # .01 is best


# CE 2ND Learning Rate Set
# initialize search parameters
cross_ent_relu_LRs_2 = np.linspace(the_best_cross_ent_relu_LR_so_far / 1.25, the_best_cross_ent_relu_LR_so_far * 20, 5)
# run the SGD loop
cross_ent_relu_LR_val_accuracy_2 = execute_parameter_search('lr', cross_ent_relu_LRs_2, cost=net.CrossEntropyCostReLU)
# plot the validation accuracy curves
plot_search_results(cross_ent_relu_LR_val_accuracy_2, cross_ent_relu_LRs_2)
# set the QC lr
the_cross_ent_relu_lr = cross_ent_relu_LRs_2[3] # 0.152 is best


# CE L2 Regularization
# CE 1ST lambdas set
cross_ent_relu_lmbda_val_accuracy_1 = execute_parameter_search('lambda', INITIAL_LAMBDAS, the_best_lr=the_cross_ent_relu_lr,
                                                          cost=net.CrossEntropyCostReLU)
# plot the validation accuracy curves
plot_search_results(cross_ent_relu_lmbda_val_accuracy_1, INITIAL_LAMBDAS)
# select the best one
the_best_cross_ent_relu_lmbda_so_far = INITIAL_LAMBDAS[2]  # .1 is best

# CE 2ND lambdas set
# initialize search parameters
cross_ent_relu_lmbdas_2 = np.linspace(the_best_cross_ent_relu_lmbda_so_far * 5, the_best_cross_ent_relu_lmbda_so_far * 30, 5)
cross_ent_relu_lmbdas_2 = np.linspace(the_best_cross_ent_relu_lmbda_so_far, the_best_cross_ent_relu_lmbda_so_far * 5, 5)
# run the SGD loop
cross_ent_relu_lmbda_val_accuracy_2 = execute_parameter_search('lambda', cross_ent_relu_lmbdas_2, the_best_lr=the_cross_ent_relu_lr,
                                                          cost=net.CrossEntropyCostReLU)
# plot the validation accuracy curves
plot_search_results(cross_ent_relu_lmbda_val_accuracy_2, cross_ent_relu_lmbdas_2)
# set the L2 value
the_cross_ent_relu_lmbda = cross_ent_relu_lmbdas_2[0]  # 0.005 is best


#### Cross Entropy Cost w/ ReLU

test_start_time = default_timer()
net_cross_ent_relu= net.Network(LAYERS, cost=net.CrossEntropyCostReLU)
CE_test_results = net_cross_ent_relu.SGD(eta=the_cross_ent_relu_lr, lmbda=the_cross_ent_relu_lmbda, **get_test_parameters())
print('\n', round((default_timer() - test_start_time) / 60, 1))

cross_ent_relu_test_results_dict = {
    'train': CE_test_results[3],
    'test': CE_test_results[1]
}

plot_test_results(cross_ent_relu_test_results_dict)
