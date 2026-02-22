from typing import Any, cast
import data_processor as dp
import linear_classifier as lc

class main:
    def __init__(self):
        url = "assets/Iris.csv"
        data_processor = dp.data_processor(cast(Any, url))
        data_processor.initialzie_data()
        data = data_processor.get_data()

        linear_classifier = lc.linear_classifier(0.01,1)
        linear_classifier.fit(data)
        print(linear_classifier.get_weights())

if __name__ == "__main__":
    main()