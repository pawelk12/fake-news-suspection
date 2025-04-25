from prepare_data import Dataset
from train_model import Model

def main():
    dataset = Dataset()
    dataset.merge_dataframes()
    dataset.clean_data()
    dataset.divide_train_test()
    model = Model()
    model.tokenize(dataset.train_texts, dataset.test_texts)
    model.prepare_for_training_w_torch(dataset.train_labels, dataset.test_labels)
    model.train()

if __name__ == "__main__":
    main()
