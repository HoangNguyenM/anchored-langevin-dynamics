import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import datasets, prior_sample, optimizers, classifiers

def main(scale=1, step_size=0.1, iterations=100):
    sample_size = 50
    batch_size = 200

    parallel = True

    data_name = 'banknote'

    X, Y = datasets.get_data(data_name = data_name)
    print(f"The shape of X is {X.shape}")
    print(f"The shape of Y is {Y.shape}")

    nnodes=32
    prior_method = 'normal'
    prior_std = 2

    # Define optimizers
    optimizers_list = []

    # Gaussian smoothing optimizers
    optimizers_list.append(optimizers.vanilla_LD_Gauss_smooth(step_size = step_size))
    optimizers_list.append(optimizers.vanilla_anchored_LD_Gauss_smooth(step_size = step_size))
    # optimizers_list.append(optimizers.vanilla_time_change_LD_Gauss_smooth(step_size = step_size))

    # Save optimizers names
    opt_num = len(optimizers_list)
    names = [optimizers_list[k].name for k in range(opt_num)]

    # get the prior weights, prior weights have shape (sample_size, nnodes, features) for layer 1 and (sample_size, 1, nnodes) for layer 2
    prior1 = np.array([prior_sample.sample_prior(sample_size=nnodes, d=X.shape[1], method=prior_method, std=prior_std) for _ in range(sample_size)])
    prior2 = np.array([prior_sample.sample_prior(sample_size=1, d=nnodes, method=prior_method, std=prior_std) for _ in range(sample_size)])

    # Define classifiers
    classifiers_list = []
    for k in range(opt_num):
        classifiers_list.append(classifiers.NN(nnodes=nnodes, x=X, y_true=Y, prior1=prior1.copy(), prior2=prior2.copy(), 
                                               optimizer=optimizers_list[k], runs=sample_size, batch_size=batch_size, scale=scale, lr=step_size))

    # Define training function for each optimizer
    def train(classifier):
        classifier.fit(epochs=iterations)
        return [classifier.accuracies, classifier.loss]
    
    if parallel:
        # Parallelize the training process by optimizers
        print("____Start training process____")
        pool = Parallel(n_jobs = opt_num, backend = 'loky', verbose = 51, pre_dispatch = 'all')
        results = pool(delayed(train)(classifier=classifiers_list[k]) for k in range(opt_num))
        results = np.array(results)
    else:
        # Run the training process without parallelization
        results = []
        for k in range(opt_num):
            print(f"Current optimizer is {names[k]}")
            classifiers_list[k].fit(epochs=iterations)
            results.append(train(classifier=classifiers_list[k]))
    
    return results[:,0], results[:,1], names


### The main code ###

scales_list = [2]
lr_list = [0.1]
iterations = 300

# For multiple scale test
accs_list = []
losses_list = []
names_list = []

for i in range(len(scales_list)):
    accs, losses, names = main(scale=scales_list[i], step_size=lr_list[i], iterations=iterations)
    accs_list.append(accs)
    losses_list.append(losses)
    names_list.append(names)

x = [i for i in range(len(accs_list[0][0]))]
for j in range(len(accs_list[0])):
    for i in range(len(scales_list)):
        plt.plot(x, accs_list[i][j], 
                    label = names_list[i][j] + ' @ scales=' + str(scales_list[i]) + ', lr=' + str(lr_list[i]))

plt.ylabel('Accuracy')
plt.xlabel('Iteration')
plt.legend()
plt.show()
plt.close()

for j in range(len(losses_list[0])):
    for i in range(len(scales_list)):
        plt.plot(x, losses_list[i][j], 
                    label = names_list[i][j] + ' @ scales=' + str(scales_list[i]) + ', lr=' + str(lr_list[i]))

plt.ylabel('Accuracy')
plt.xlabel('Iteration')
plt.legend()
plt.show()