import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import prior_sample, datasets, optimizers, regularizers, classifiers

def run(sample_size = 50, simulation_num = 200, 
        scale = 1, step_size = 0.1, iterations = 1000,
        data_name = 'cancer', reg_name = 'Lasso',
        coef = 1, a = 10, varep = 0.5, l2_coef = 0.5,
        parallel = True):
    
    """Run logistic regression
    Args:
        sample_size: the number of regressions run to be averaged over
        simulation_num: the number of Monte Carlo simulation for Gaussian smoothing
        data_name: cancer or banknote
        reg_name: regularizer, can be 'Lasso', 'SCAD' or 'MCP'
        coef, a: hyperparameters of the regularizer
        varep, l2_coef: hyperparameters for deterministic smoothing
        parallel: if True, run parallel over CPU cores
    """

    X, Y = datasets.get_data(data_name = data_name)
    print(f"The shape of X is {X.shape}")
    print(f"The shape of Y is {Y.shape}")

    # Define regularizers
    fn_value = regularizers.get_regularizer(reg_name = reg_name, coef = coef, a = a)

    ref_value = regularizers.get_ref_regularizer(reg_name = reg_name, coef = coef, a = a, varep = varep)

    grad_value = regularizers.get_grad_regularizer(reg_name = reg_name, coef = coef, a = a, varep = varep, l2_coef = l2_coef)

    # Define optimizers
    optimizers_list = []

    # Gaussian smoothing optimizers
    optimizers_list.append(optimizers.LD_Gauss_smooth(fn_value=fn_value, 
                                                    simulation_num=simulation_num, scale=scale, step_size=step_size))
    optimizers_list.append(optimizers.anchored_LD_Gauss_smooth(fn_value=fn_value, 
                                                    simulation_num=simulation_num, scale=scale, step_size=step_size))
    # uncomment to add time change Langevin optimizer
    #optimizers_list.append(optimizers.time_change_LD_Gauss_smooth(fn_value=fn_value, 
    #                                                simulation_num=simulation_num, scale=scale, step_size=step_size))

    # Deterministic smoothing optimizers
    optimizers_list.append(optimizers.LD_reg_smooth(fn_value=fn_value, 
                                                    ref_value=ref_value, grad_value=grad_value, step_size=step_size))
    optimizers_list.append(optimizers.anchored_LD_reg_smooth(fn_value=fn_value, 
                                                    ref_value=ref_value, grad_value=grad_value, step_size=step_size))
    # uncomment to add time change Langevin optimizer
    #optimizers_list.append(optimizers.time_change_LD_reg_smooth(fn_value=fn_value, 
    #                                                ref_value=ref_value, grad_value=grad_value, step_size=step_size))
    # Save optimizers names
    opt_num = len(optimizers_list)
    names = [optimizers_list[k].name for k in range(opt_num)]

    # Make prior sample data
    prior_weight = prior_sample.sample_prior(sample_size=sample_size, d=X.shape[1], method='laplace', std=2)

    # Define classifiers
    classifiers_list = []
    for k in range(opt_num):
        classifiers_list.append(classifiers.logistic_regressor(x=X, y_true=Y, 
                                                               prior=prior_weight.copy(), optimizer=optimizers_list[k]))

    
    # Define training function for each optimizer
    def train(classifier):
        classifier.fit(epochs=iterations)
        return [classifier.accuracies, classifier.std]
    
    if parallel:
        # Parallelize the training process by optimizers
        print("____Start training process____")
        pool = Parallel(n_jobs = opt_num, backend = 'loky', verbose = 51, pre_dispatch = 'all')
        results = pool(delayed(train)(classifier=classifiers_list[k]) for k in range(opt_num))
        results = np.array(results)
        accs = results[:,0,:]
        stds = results[:,1,:]
    else:
        # Run the training process without parallelization
        accs = []
        stds = []
        for k in range(opt_num):
            print(f"Current optimizer is {names[k]}")
            classifiers_list[k].fit(epochs=iterations)
            result = train(classifier=classifiers_list[k])
            accs.append(result[0])
            stds.append(result[1])
    
    return accs, stds, names


### The main code ###
if __name__ == "__main__":
    # Define hyperparameters
    sample_size = 40
    simulation_num = 20

    data_name = 'cancer'

    # coefficients for defining the regularizer
    coef = 10
    a = 10

    # coefficients for deterministic smoothing
    varep = 0.5
    l2_coef = 0.5

    # regularizer can be 'Lasso', 'SCAD' or 'MCP'
    reg_name = 'Lasso'

    scales_list = [0.5, 1]
    lr_list = [0.01, 0.02]
    iterations = 100

    # For single scale test
    if len(scales_list) < 2:
        results, stds, names = run(sample_size=sample_size, simulation_num=simulation_num, 
                                   scale=scales_list[0], step_size=lr_list[0], iterations=iterations,
                                   data_name=data_name, reg_name=reg_name,
                                   coef=coef, a=a, varep=varep, l2_coef=l2_coef)

        x = [i for i in range(len(results[0]))]
        plt.fill_between(x, results[0]-stds[0], results[0]+stds[0], color='paleturquoise')
        plt.fill_between(x, results[1]-stds[1], results[1]+stds[1], color='lightblue', label='Accuracy \u00B1 std')
        plt.plot(x, results[0], label = names[0])
        plt.plot(x, results[1], label = names[1], color = 'b')
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Iteration')
        plt.legend()
        plt.show()

    # For multiple scale test
    else:
        results_list = []
        names_list = []

        for i in range(len(scales_list)):
            results, _, names = run(sample_size=sample_size, simulation_num=simulation_num, 
                                    scale=scales_list[i], step_size=lr_list[i], iterations=iterations,
                                    data_name=data_name, reg_name=reg_name,
                                    coef=coef, a=a, varep=varep, l2_coef=l2_coef)
            results_list.append(results)
            names_list.append(names)

        for j in range(len(results_list[0])):
            for i in range(len(scales_list)):
                plt.plot([i for i in range(len(results_list[i][j]))], results_list[i][j], 
                        label = names_list[i][j] + ' @ scales=' + str(scales_list[i]) + ', lr=' + str(lr_list[i]))

        plt.ylabel('Accuracy (%)')
        plt.xlabel('Iteration')
        plt.legend()
        plt.show()