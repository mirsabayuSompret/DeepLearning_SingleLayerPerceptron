
import pandas as pd
#class used to process data from url,
#would give the data for training and testing the model
class data_processor:
    def __init__(self, url: str):
        self.url = url
        self.data_frame = pd.DataFrame()
        self.data_train = pd.DataFrame()
        self.data_test = pd.DataFrame()

    def initialzie_data(self):
        self.data_frame = pd.read_csv(self.url)
        if not self.data_frame.notnull : 
            return
        
        self.__clean_data()
        self.__drop_unnecessary_columns()
        self.__drop_unnecessary_rows_based_on_value("Iris-virginica")
        self.__change_target_column_to_binary()
        self.__set_data_training_and_testing()
        pass

    def __clean_data(self):
        self.data_frame.dropna(inplace=True)
        self.data_frame.drop_duplicates(inplace=True)
        pass

    def __drop_unnecessary_columns(self):
        self.data_frame.drop(columns=['Id'], inplace=True)
        pass

    def __drop_unnecessary_rows_based_on_value(self, value):
        self.data_frame = self.data_frame[self.data_frame['Species'] != value]
        pass

    def __change_target_column_to_binary(self):
        self.data_frame['Species'] = self.data_frame['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1})
        pass

    def __set_data_training_and_testing(self):
        # get data for target 0,
        data_target_0 = self.data_frame[self.data_frame['Species'] == 0]
        # get data for target 1,
        data_target_1 = self.data_frame[self.data_frame['Species'] == 1]

        x_train_data_0 = data_target_0.iloc[:-10].reset_index(drop=True)
        x_test_data_0 = data_target_0.iloc[-10:].reset_index(drop=True)
        x_train_data_1 = data_target_1.iloc[:-10].reset_index(drop=True)
        x_test_data_1 = data_target_1.iloc[-10:].reset_index(drop=True)

        self.data_train = pd.concat([x_train_data_0, x_train_data_1]).reset_index(drop=True)
        self.data_test = pd.concat([x_test_data_0, x_test_data_1]).reset_index(drop=True)

    def get_data_train(self):
        return self.data_train
    
    def get_data_test(self):
        return self.data_test
