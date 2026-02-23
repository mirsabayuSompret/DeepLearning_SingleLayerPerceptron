import numpy as np
from sklearn.model_selection import train_test_split

class linear_classifier:
    def __init__(self, learning_rate=0.01, epochs=5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = []
        self.bias = None

    

    def fit(self, data_train, theta = [0.2,0.2,0.2,0.2], bias = 0.2):
        self.theta = np.array(theta)
        self.bias = bias

        x_train = data_train.drop(columns=['Species'])
        y_train = data_train['Species'].values
        check_column = np.zeros(x_train.shape[0])
        y_predicted = np.zeros(x_train.shape[0])
        error = np.zeros(x_train.shape[0])

        for epoch in range(self.epochs):
            for index, row in x_train.iterrows():
                hypothesis = self.__hypothesis_function(row, self.theta, self.bias)
                sigmoid = self.__sigmoid(hypothesis)
                delta_theta = 2*(sigmoid - y_train[index]) * (1-sigmoid) *(sigmoid * row)
                delta_bias = 2*(sigmoid - y_train[index]) * (1-sigmoid) * sigmoid

                y_predicted[index] = 1 if sigmoid >= 0.5 else 0
                error[index] = np.pow(sigmoid - y_train[index], 2)

                check_column[index] = 1 if y_predicted[index] == y_train[index] else 0
                self.theta = self.theta - self.learning_rate * delta_theta
                self.bias = self.bias - self.learning_rate * delta_bias

    def __sigmoid(self, value):
        return 1 / (1 + np.exp(-value))
    
    def __hypothesis_function(self, X, theta, bias):
        return X.dot(theta) + bias

    def get_theta(self):
        return self.theta
    
    def get_bias(self):
        return self.bias

    def predict(self, X):
        hypothesis = self.__hypothesis_function(X, self.theta, self.bias)
        sigmoid = self.__sigmoid(hypothesis)
        return np.where(sigmoid >= 0.5, 1, 0)
        