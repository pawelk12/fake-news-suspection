from prepare_data import Dataset

def main():
    dataset = Dataset()
    dataset.merge_dataframes()
    dataset.clean_data()
    dataset.divide_train_test()

if __name__ == "__main__":
    main()
