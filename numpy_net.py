# neural network with 3 layers
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def sigmoid(vec):
    """ Sigmoid activation function

    Args:
        vec: input features vector
    """
    return 1/(1 + np.exp(-vec))


def diff_sigmoid(vec):
    """ derivative of the sigmoid 
    
    Args:
        vec: vector with respect to differentiate for
    """
    return sigmoid(vec) * (1. - sigmoid(vec))


class network(object):
    def __init__(self, n_features, alpha=0.01):
        """ Initalize weights and biases of the neural net """
        self.alpha = alpha

        self.W1 = np.random.rand(10, n_features) * 0.01
        self.W2 = np.random.rand(5, 10) * 0.01
        self.W3 = np.random.rand(1, 5) * 0.01
        
        self.b1 = np.ones((10, 1))
        self.b2 = np.ones((5, 1))
        self.b3 = np.ones((1, 1))

        # "memory" of the network
        # we keep intermediate results needed for the backpropagation

        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
        self.z3 = None
        self.a3 = None        

    @staticmethod
    def cross_entropy_loss(y_hat, y):
        """ calculates the loss function 
        
            Args:
                y_hat: calculated output array
                y:     ground truth array      
        """
        y_hat = np.squeeze(y_hat)
        loss = y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
        return - loss

    def forward_prop(self, x):
        """ Calculates the output of the neural net

        Args:
            x: input features vector
        """
        self.x = x
        self.z1 = np.dot(self.W1, x) + self.b1
        self.a1 = sigmoid(self.z1)

        self.z2 = np.dot(self.W2, self.a1) + self.b2
        self.a2 = sigmoid(self.z2)

        self.z3 = np.dot(self.W3, self.a2) + self.b3
        self.a3 = sigmoid(self.z3)  # output of the network

        return self.a3
    
    def backprop(self, y_hat, y):
        """ backpropagation """
        self.dCdz3 = y_hat - y
        self.dCdW3 = np.dot(self.dCdz3, self.a2.T)
        self.dCdb3 = self.dCdz3

        self.dCdz2 = np.dot(self.W3.T, self.dCdz3) * diff_sigmoid(self.z2)
        self.dCdW2 = np.dot(self.dCdz2, self.a1.T)
        self.dCdb2 = self.dCdz2

        self.dCdz1 = np.dot(self.W2.T, self.dCdz2) * diff_sigmoid(self.z1)
        self.dCdW1 = np.dot(self.dCdz1, self.x.T)
        self.dCdb1 = self.dCdz1        

        self.update()

    def update(self):
        """ update the weights and biases """
        self.W1 = self.W1 - self.alpha * np.squeeze(self.dCdW1)
        self.W2 = self.W2 - self.alpha * self.dCdW2
        self.W3 = self.W3 - self.alpha * self.dCdW3

        self.b1 = self.b1 - self.alpha * self.dCdb1
        self.b2 = self.b2 - self.alpha * self.dCdb2
        self.b3 = self.b3 - self.alpha * self.dCdb3


if __name__ == "__main__":

    # iris = datasets.load_iris()
    # df = np.c_[iris.data, iris.target]
    # np.random.shuffle(df)


    # X_train, X_test, y_train, y_test = train_test_split(df[:, :-1], df[: , -1], test_size = 0.3)

    dataset = pd.read_csv("seeds_dataset2.csv", delimiter=";", header=0)
    shuffled_dataset = dataset.sample(frac=1).reset_index(drop=True)
    shuffled_dataset['Class'] = shuffled_dataset['Class'] - 1

    X = shuffled_dataset.iloc[:, 0:-1].values
    y = shuffled_dataset.iloc[:, -1].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    the_net = network(alpha=0.2, n_features=X_train.shape[1])

    epoch_loss = []
    for epoch in range(0, 20):
        # train and update one sample at a time
        for sample in range(X_train.shape[0]):
            x = np.expand_dims(X_train[sample, :], axis=1)
            # forward pass
            y_hat = the_net.forward_prop(x)
            # compute the loss
            loss = the_net.cross_entropy_loss(y_hat, y_train[sample])   
            # backpropagate and update weights and biases
            the_net.backprop(y_hat, y_train[sample])

            # print(1)

        print("Epoch = {}, loss = {}".format(epoch, loss))
        epoch_loss.append(loss)

    plt.figure()
    plt.plot(epoch_loss, marker=".")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.show()
    print("end")