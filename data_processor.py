
import pandas as pd
#class used to process data from url,
#would give the data for training and testing the model
class data_processor:
    def __init__(self, url: str):
        self.url = url
        self.data_frame = pd.DataFrame()

    def initialzie_data(self):
        self.data_frame = pd.read_csv(self.url)
        if not self.data_frame.notnull : 
            return
        
        self.__clean_data()
        self.__drop_unnecessary_columns()
        self.__drop_unnecessary_rows_based_on_value("Iris-virginica")
        self.__change_target_column_to_binary()
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

    def get_data(self):
        return self.data_frame
    
    
