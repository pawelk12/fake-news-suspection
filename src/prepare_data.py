import pandas as pd


class Dataset:
    def __init__(self):
        try:
            self.fake_news_frame = pd.read_csv("../dataset/Fake.csv")
            self.true_news_frame = pd.read_csv("../dataset/True.csv")
        except FileNotFoundError as e:
            print(e)
