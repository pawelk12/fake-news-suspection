import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

class Dataset:
    def __init__(self):
        self.frame = None
        try:
            self.fakes_frame = pd.read_csv("../dataset/Fake.csv")
            # print(self.fakes_frame.isnull().sum())
            self.reals_frame = pd.read_csv("../dataset/True.csv")
            # print(self.reals_frame.isnull().sum())
        except FileNotFoundError as e:
            print(e)

    def merge_dataframes(self):
        self.fakes_frame['label'] = 0
        self.reals_frame['label'] = 1
        frames = [self.fakes_frame, self.reals_frame]
        self.frame = pd.concat(frames, ignore_index=True)


    def clean_data(self):
        # print(self.frame.isnull().sum())  ==> 0
        self.frame.drop_duplicates(inplace=True)
        self.frame.drop('date', axis=1, inplace=True)
        self.frame.drop('subject', axis=1, inplace=True)

        self.frame['content'] = self.frame['title'] + ' ' + self.frame['text']
        self.frame.drop('title', axis=1, inplace=True)
        self.frame.drop('text', axis=1, inplace=True)
        

    def divide_train_test(self):
        self.train_texts, self.test_texts, self.train_labels, self.test_labels = train_test_split(
            self.frame["content"].tolist(), self.frame["label"].tolist(), test_size=0.2, random_state=42)



        
