import numpy as np
import

class linear_classifier:
    def __init__(self, learning_rate=0.01, epochs=5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight = []
        self.bias = None

    

    def fit(self, data, weight = [0.2,0.2,0.2,0.2], bias = 0.2):
        self.weight = np.array(weight)
        self.bias = bias

        X_train, y_train, X_test, y_test = self.__split_data(data)

        X = data.drop(columns=['Species'])
        y = data['Species']

        check_column = np.zeros(X.shape[0])
        y_predicted = np.zeros(X.shape[0])
        error = np.zeros(X.shape[0])
        for epoch in range(self.epochs):
            for index, row in X.iterrows():
                hypothesis = self.__hypothesis_function(row, self.weight, self.bias)
                sigmoid = self.__sigmoid(hypothesis)
                delta_weight = 2*(sigmoid - y[index]) * (1-sigmoid) *(sigmoid * row)
                delta_bias = 2*(sigmoid - y[index]) * (1-sigmoid) * sigmoid
                y_predicted[index] = 1 if sigmoid >= 0.5 else 0
                error[index] = np.pow(sigmoid - y[index], 2)
                check_column[index] = 1 if y_predicted[index] == y[index] else 0
                self.weight -= self.learning_rate * delta_weight
                self.bias -= self.learning_rate * delta_bias

    def __sigmoid(self, value):
        return 1 / (1 + np.exp(-value))
    
    def __hypothesis_function(self, X, weight, bias):
        return X.dot(weight) + bias

    def get_weights(self):
        return self.weight
    
    def get_bias(self):
        return self.bias

    def predict(self, X):
        pass
        