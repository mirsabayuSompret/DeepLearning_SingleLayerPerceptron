import numpy as np
from sklearn.model_selection import train_test_split

class linear_classifier:
    def __init__(self, learning_rate=0.01, epochs=5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = [0.2,0.2,0.2,0.2]
        self.bias = 0.2

    def fit(self, data_train):

        x_train = data_train.drop(columns=['Species'])
        y_train = data_train['Species'].values

        check_column = np.zeros(x_train.shape[0])
        y_predicted = np.zeros(x_train.shape[0])
        error = np.zeros(x_train.shape[0])
        mse = 0
        accuracy = 0
        for index, row in x_train.iterrows():
            hypothesis = self.__hypothesis_function(row, self.theta, self.bias)
            sigmoid = self.__sigmoid(hypothesis)
            delta_theta = 2*(sigmoid - y_train[index]) * (1-sigmoid) * (sigmoid * row.values)
            delta_bias = 2*(sigmoid - y_train[index]) * (1-sigmoid) * sigmoid

            y_predicted[index] = 1 if sigmoid >= 0.5 else 0
            error[index] = np.pow(sigmoid - y_train[index], 2)

            check_column[index] = 1 if y_predicted[index] == y_train[index] else 0
            self.theta = self.theta - self.learning_rate * delta_theta
            self.bias = self.bias - self.learning_rate * delta_bias

        mse = np.mean(error)
        accuracy = (np.sum(check_column) / x_train.shape[0])
        return mse, accuracy
    
    def predict(self, data, theta, bias):
        x_test = data.drop(columns=['Species'])
        y_test = data['Species'].values

        check_column = np.zeros(x_test.shape[0])
        y_predicted = np.zeros(x_test.shape[0])
        error = np.zeros(x_test.shape[0])
        mse = 0
        accuracy = 0

        for index, row in x_test.iterrows():
            hypothesis = self.__hypothesis_function(row, theta, bias)
            sigmoid = self.__sigmoid(hypothesis)

            y_predicted[index] = 1 if sigmoid >= 0.5 else 0
            error[index] = np.pow(sigmoid - y_test[index], 2)

            check_column[index] = 1 if y_predicted[index] == y_test[index] else 0

        mse = np.mean(error)
        accuracy = (np.sum(check_column) / x_test.shape[0])
        return mse, accuracy


    def __sigmoid(self, value):
        return 1 / (1 + np.exp(-value))
    
    def __hypothesis_function(self, X, theta, bias):
        return np.dot(X, theta) + bias

    def get_theta(self):
        return self.theta
    
    def get_bias(self):
        return self.bias
    

        