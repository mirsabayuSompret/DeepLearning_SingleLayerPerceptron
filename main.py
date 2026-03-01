from typing import Any, cast
import data_processor as dp
import linear_classifier as lc

class main:
    def __init__(self):
        url = "assets/Iris.csv"
        data_processor = dp.data_processor(cast(Any, url))
        data_processor.initialzie_data()
        data_train = data_processor.get_data_train()
        data_test = data_processor.get_data_test()
        # print("data_train", data_train.head())
        # print("data_test", data_test.head())

        linear_classifier = lc.linear_classifier(0.1,1)
        mse_train, accuracy_train = linear_classifier.fit(data_train)
        # print(linear_classifier.get_theta())
        # print(linear_classifier.get_bias())
        # print(f"Mean Squared Error train: {mse_train}")
        # print(f"Accuracy train: {accuracy_train}%")

        
        mse, accuracy = linear_classifier.predict(data_test, linear_classifier.get_theta(), linear_classifier.get_bias())

        print(f"Mean Squared Error test: {mse}")
        print(f"Accuracy test: {accuracy}%")

if __name__ == "__main__":
    main()