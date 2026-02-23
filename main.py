from typing import Any, cast
import data_processor as dp
import linear_classifier as lc

class main:
    def __init__(self):
        url = "assets/Iris.csv"
        data_processor = dp.data_processor(cast(Any, url))
        data_processor.initialzie_data()
        data = data_processor.get_data_train()

        linear_classifier = lc.linear_classifier(0.1,1)
        linear_classifier.fit(data)
        print(linear_classifier.get_theta())
        print(linear_classifier.get_bias())

if __name__ == "__main__":
    main()