from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        self.path = 'data.txt'
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
        x_data = list()
        y_data = list()
        for text in full_text.splitlines():
            x_data.append('<BEGIN>' + text)
            y_data.append(text + '<END>')
        return x_data, y_data, full_text





