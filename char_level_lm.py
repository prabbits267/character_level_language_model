import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader


from my_dataset import *
dataset = MyDataset()

full_text = dataset.text

tokens = sorted(set(full_text))

token_to_ind = {w:i for i,w in enumerate(tokens)}
ind_to_token = {token_to_ind[w]:w for w in token_to_ind}
P = 0

vocab_size = len(tokens)
use_cuda = torch.cuda.is_available()
device = ('cuda:0' if use_cuda else 'cpu')


BATCH_SIZE = 64
EMBED_SIZE = 200
HIDDEN_SIZE = 64
OUTPUT_SIZE = vocab_size
NUM_LAYERS = 2

dataloader = DataLoader(dataset=dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True)


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, output_size, embed_size, num_layers):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            input_size = embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.softmax = nn.Softmax()

    def forward(self, input, input_len):
        batch_size, seq_len = input.size()
        packed_input = pack_padded_sequence(input, input_len)
        output, hidden_state, cell_state = self.lstm(packed_input)
        unpacked_output, _ = pad_packed_sequence(output)
        out = unpacked_output.view(-1, unpacked_output.size(2))
        out = self.softmax(out)
        out = out.view(batch_size, seq_len, vocab_size)
        return out

    def create_variable(self,tensor):
        return Variable(tensor.to(device))

    def sent_to_seq(self, input):
        return torch.LongTensor([token_to_ind[w] for w in input])

    def pad_sequence(self, input):
        max_len = len(input[0])
        batch_size = len(input)
        input_seq = torch.zeros(batch_size, max_len).long()
        for i, seq in enumerate(input):
            input_seq[i, :len(seq)] = self.sent_to_seq(seq)
        return input_seq

    def create_batch(self, input, output):
        input_seq = [list(w) for w in list(input)]
        target_seq = [list(w) for w in list(output)]

        seq_pairs = sorted(zip(input_seq, target_seq), key=lambda p: len(p[0]), reverse=True)
        input_seq, target_seq = zip(*seq_pairs)

        input_seq = self.create_batch_tensor(input_seq)
        target_seq = self.create_batch_tensor(target_seq)
        return input_seq, target_seq

    def create_batch_tensor(self, input):
        input_seq = self.pad_sequence(input)
        return input_seq



# def convert
lm = LanguageModel(HIDDEN_SIZE, OUTPUT_SIZE, EMBED_SIZE, NUM_LAYERS)
for i, (x_data, y_data) in enumerate(dataloader):
    input, output = lm.create_batch(x_data, y_data)
    print(input)
    print(output)




