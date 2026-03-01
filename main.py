from typing import Any, cast
import data_processor as dp
import linear_classifier as lc
import numpy as np
import matplotlib.pyplot as plt

class main:
    def __init__(self):
        url = "assets/Iris.csv"
        data_processor = dp.data_processor(cast(Any, url))
        data_processor.initialzie_data()
        data_train = data_processor.get_data_train()
        data_test = data_processor.get_data_test()
        
        #initial parameters for training the model
        epochs = 5
        learning_rate = 0.1
        initial_theta = [0.2, 0.2, 0.2, 0.2]
        initial_bias = 0.2

        # 2d array to store mse and accuracy for training and testing data,
        # 0 for training data, 1 for testing data, and columns for each epoch
        mse_points = np.zeros((2, epochs))
        accuracy_points = np.zeros((2, epochs))

        linear_classifier = lc.linear_classifier(learning_rate, initial_theta, initial_bias)

        for epoch in range(epochs):
            mse_train, accuracy_train = linear_classifier.fit(data_train)
            mse_test, accuracy_test = linear_classifier.predict(data_test, linear_classifier.get_theta(), linear_classifier.get_bias())
            mse_points[0][epoch] = mse_train
            mse_points[1][epoch] = mse_test
            accuracy_points[0][epoch] = accuracy_train
            accuracy_points[1][epoch] = accuracy_test

        print(f"mse_points: {mse_points}")
        print(f"accuracy_points: {accuracy_points}")

        #create diagrams for mse and accuracy for training and testing data
        x = np.arange(1, epochs + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # MSE plot
        axes[0].plot(x, mse_points[0], marker="o", label="Train MSE")
        axes[0].plot(x, mse_points[1], marker="o", label="Test MSE")
        axes[0].set_title("MSE per Epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE")
        axes[0].set_xticks(x)
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend()

        for i, v in enumerate(mse_points[0]):
            axes[0].annotate(f"{v:.4f}", (x[i], v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        for i, v in enumerate(mse_points[1]):
            axes[0].annotate(f"{v:.4f}", (x[i], v), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8)

        # Accuracy plot
        axes[1].plot(x, accuracy_points[0], marker="o", label="Train Accuracy")
        axes[1].plot(x, accuracy_points[1], marker="o", label="Test Accuracy")
        axes[1].set_title("Accuracy per Epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_xticks(x)
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].legend()

        for i, v in enumerate(accuracy_points[0]):
            axes[1].annotate(f"{v:.4f}", (x[i], v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        for i, v in enumerate(accuracy_points[1]):
            axes[1].annotate(f"{v:.4f}", (x[i], v), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8)

        plt.tight_layout()
        plt.show()


        

if __name__ == "__main__":
    main()