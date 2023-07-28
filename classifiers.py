import numpy as np
import prior_sample

class logistic_regressor():
    """Define logistic regression classifier
    Args:
        x: the data, have size (observation_size, features)
        y_true: the labels, have size (observation_size,), reshaped into (sample_size, observation_size)
        prior: the prior distribution of the weights, have size (sample_size, features)
        optimizer: the optimizer
    """
    def __init__(self, x, y_true, prior, optimizer):
        self.x = x
        self.y_true = np.array([y_true] * prior.shape[0])
        self.weights = prior
        self.optimizer = optimizer
        self.accuracies = []
        self.std = []
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(np.minimum(-x, np.full(x.shape, 500))))
    
    def compute_gradients(self, y_pred):
        # get the gradient of the binary cross entropy (differentiable part)
        diff =  y_pred - self.y_true
        grad_differentiable = np.matmul(diff, self.x) # (sample_size, features)

        # get the gradient of the regularizer (non-differentiable part)
        grad_non_differentiable = self.optimizer.get_ref_grad(self.weights) # (sample_size, features)

        return grad_differentiable + grad_non_differentiable
    
    def update_parameters(self, grad):
        self.weights += self.optimizer.update_step(self.weights, grad)

    def fit(self, epochs):
        # Start training
        for i in range(epochs):
            # get probability prediction
            x_dot_weights = np.matmul(self.weights, self.x.transpose())
            y_pred = self.sigmoid(x_dot_weights)

            # convert probability to class: p > 0.5 to class 1 and class 0 otherwise
            class_pred = np.maximum(np.sign(y_pred - 0.5), np.full(y_pred.shape, 0))

            # get accuracy
            new_accuracies = np.sum(np.abs(self.y_true - class_pred), axis = -1)/self.x.shape[0]
            self.accuracies.append(np.mean(100-new_accuracies*100))
            self.std.append(np.std(100-new_accuracies*100))
            #print(self.accuracies[i])

            # update parameters
            total_grad = self.compute_gradients(y_pred=y_pred)
            self.update_parameters(total_grad)

class NN_layer():
    """Define neural network layer
    Args:
        weight has shape (sample_size, nnodes, features)
    """
    def __init__(self, nnodes, activation, prior):
        self.nnodes = nnodes
        self.weight = prior
        self.activation = activation

class NN():
    """Define 2-layer neural network classifier
    Args:
        x: the data, have size (observation_size, features)
        y_true: the labels, have size (observation_size,), reshaped into (sample_size, observation_size)
        prior weights have size (sample_size, nnodes, features) for layer 1 and (sample_size, 1, nnodes) for layer 2
        optimizer: the optimizer
    """
    def __init__(self, nnodes, x, y_true, prior1, prior2, optimizer, runs = 20, simulation_num = 500, scale = 1, lr = 0.1):
        self.nnodes = nnodes
        self.lr = lr
        self.x = x
        self.y_true = np.array([y_true] * runs)
        self.optimizer = optimizer
        self.simulation_num = simulation_num
        self.scale = scale
        
        self.L1 = NN_layer(nnodes=nnodes, activation=self.relu, prior=prior1)
        self.L2 = NN_layer(nnodes=1, activation=self.sigmoid, prior=prior2)

        self.accuracies = []
        self.loss = []

    def sigmoid(self, x):
        return 1/(1 + np.exp(np.minimum(-x, np.full(x.shape, 500))))
    
    def relu(self, x):
        return np.maximum(x, np.full(x.shape, 0))
    
    def relu_grad(self, x):
        return np.maximum(np.sign(x), np.full(x.shape, 0))
    
    # run through the neural network
    # w1 has shape (simulation_num, sample_size, nnodes, features)
    # w2 has shape (simulation_num, sample_size, 1, nnodes)
    # output has shape (simulation_num, sample_size, observation_size)
    def forward(self, w1, w2):
        relu_input = np.matmul(w1, self.x.transpose())
        relu_output = self.L1.activation(relu_input) # (simulation_num, sample_size, nnodes, observation_size)
        sigmoid_input = np.matmul(w2, relu_output) # (simulation_num, sample_size, 1, observation_size)
        probability = self.L2.activation(sigmoid_input[:,:,0,:])
        return probability
    
    # compute cross entropy loss
    # y_pred has shape (simulation_num, sample_size, observation_size)
    # output has shape (simulation_num, sample_size)
    def compute_loss(self, y_pred):
        y_true = np.array([self.y_true] * y_pred.shape[0])
        y_zero_loss = np.real(y_true * np.log(y_pred + 1e-9))
        y_one_loss = np.real((1-y_true) * np.log(1 - y_pred + 1e-9))
        return -np.mean(y_zero_loss + y_one_loss, axis=-1)

    def compute_gradients(self):
        # Use Gaussian smoothing to approximate the gradients of L1 weight
        noise1 = np.random.normal(0,1,tuple([self.simulation_num]) + self.L1.weight.shape) # (simulation_num, sample_size, nnodes, features)
        probability1 = self.forward(w1 = np.array([self.L1.weight]*self.simulation_num)+self.scale*noise1, w2 = np.array([self.L2.weight]*self.simulation_num))
        loss_batch1 = self.compute_loss(probability1)

        grad1 = np.mean(loss_batch1[...,None,None] * noise1, axis=0)/self.scale
        ref_loss1 = np.mean(loss_batch1, axis=0)

        # # Use Gaussian smoothing to approximate the gradients of L2 weight
        # noise2 = np.random.normal(0,1,tuple([self.simulation_num]) + self.L2.weight.shape) # (simulation_num, sample_size, 1, nnodes)
        # probability2 = self.forward(w1 = np.array([self.L1.weight]*self.simulation_num), w2 = np.array([self.L2.weight]*self.simulation_num)+self.scale*noise2)
        # loss_batch2 = self.compute_loss(probability2)

        # grad2 = np.mean(loss_batch2[...,None,None] * noise2, axis=0)/self.scale
        # ref_loss2 = np.mean(loss_batch2, axis=0)

        # Get the true gradient of L2 weight
        ref_loss2 = 0
        relu_input = np.matmul(self.L1.weight, self.x.transpose()) # (sample_size, nnodes, observation_size)
        relu_output = self.L1.activation(relu_input) # (sample_size, nnodes, observation_size)
        sigmoid_input = np.matmul(self.L2.weight, relu_output) # (sample_size, 1, observation_size)
        sigmoid_output = self.L2.activation(sigmoid_input[:,0,:]) # (sample_size, observation_size)
        loss_grad = sigmoid_output - self.y_true # (sample_size, observation_size)
        sigmoid_grad = sigmoid_output * (1 - sigmoid_output) # (sample_size, observation_size)
        grad2 = np.transpose(np.matmul(relu_output, (loss_grad * sigmoid_grad)[...,None]), (0, 2, 1)) # (sample_size, 1, nnodes)

        # # Get the true gradient of L1 weight
        # ref_loss1 = 0
        # relu_grad = self.relu_grad(relu_input) # (sample_size, nnodes, observation_size)
        # weighted_relu_grad = np.transpose(np.array([self.L2.weight[:,0,:]] * relu_grad.shape[-1]), (1, 2, 0)) * relu_grad # (sample_size, nnodes, observation_size)
        # grad1 = np.matmul(np.transpose(np.array([loss_grad * sigmoid_grad] * self.nnodes), (1, 0, 2)) * weighted_relu_grad,
        #                   self.x)  # (sample_size, nnodes, features)
        
        return grad1, grad2, ref_loss1, ref_loss2
    
    def update_parameters(self, grad1, grad2, ref_loss1, ref_loss2, loss):
        self.L1.weight += self.optimizer.update_step(self.L1.weight, grad1, loss, ref_loss1)
        #self.L2.weight += self.optimizer.update_step(self.L2.weight, grad2, loss, ref_loss2)
        #self.L1.weight += - self.lr * grad1
        self.L2.weight += - self.lr * grad2 + np.random.normal(0,1,grad2.shape) * (2*self.lr)**0.5

    def fit(self, epochs):
        # Start training
        for i in range(epochs):
            # run forward through the neural network
            y_pred = self.forward(self.L1.weight[None,...], self.L2.weight[None,...])[0,...]

            # compute the loss function
            loss = self.compute_loss(y_pred[None,...])[0,...]
            self.loss.append(np.mean(loss))

            # convert probability to class: p > 0.5 to class 1 and class 0 otherwise
            class_pred = np.maximum(np.sign(y_pred - 0.5), np.full(y_pred.shape, 0))

            # get accuracy
            new_accuracies = np.sum(np.abs(self.y_true - class_pred), axis = -1)/self.x.shape[0]
            self.accuracies.append(np.mean(100-new_accuracies*100))
            #print(self.accuracies[i])

            # update parameters
            grad1, grad2, ref_loss1, ref_loss2 = self.compute_gradients()
            self.update_parameters(grad1, grad2, ref_loss1, ref_loss2, loss)