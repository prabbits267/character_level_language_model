from math import floor

from torch.utils.data import Dataset


# 0.8 for training and 0.2 for testting
class MyDataset(Dataset):
    def __init__(self, is_test, split_ratio=0.8):
        self.path = 'data.txt'
        self.is_test = is_test
        self.split_ratio = split_ratio
        self.x_data, self.y_data, self.text = self.read_data()
        self.len = len(self.x_data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def read_data(self):
        file_reader = open(self.path, 'rt', encoding='utf-8')
        full_text = file_reader.read()
        file_reader.close()
        full_data_len = len(full_text.splitlines())
        train_size = floor(full_data_len * self.split_ratio)
        test_size = full_data_len - train_size
        x_data = list()
        y_data = list()
        for text in full_text.splitlines():
            x_data.append('B' + text)
            y_data.append(text + 'E')
        if self.is_test:
            return x_data[train_size:], y_data[train_size:], full_text
        return x_data[:train_size], y_data[:train_size], full_text

    def read_data_txt(self):
        with open('data_.txt', 'rt', encoding='utf-8') as file_reader:
            full_text = file_reader.read()
        list_text = full_text.splitlines()
        x_data = list()
        y_data = list()










