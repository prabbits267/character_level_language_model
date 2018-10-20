import math
from math import floor

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, NLLLoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from my_dataset import *
import re


dataset = MyDataset(is_test=False)
test_data = MyDataset(is_test=True)

data_size = dataset.len
full_text = dataset.text

tokens = sorted(set(full_text))


token_to_ind = {w: i for i, w in enumerate(tokens)}
ind_to_token = {token_to_ind[w]: w for w in token_to_ind}
P = 0

vocab_size = len(tokens)
use_cuda = torch.cuda.is_available()
device = ('cuda:0' if use_cuda else 'cpu')

BATCH_SIZE = 64
EMBED_SIZE = 200
HIDDEN_SIZE = 64
OUTPUT_SIZE = vocab_size
NUM_LAYERS = 2
NUM_EPOCH = 100
LEARNING_RATE = 0.005

num_interations = floor(data_size/BATCH_SIZE)

dataloader = DataLoader(dataset=dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True)

test_loader = DataLoader(dataset=test_data,
                         batch_size=256)


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, output_size, embed_size, num_layers):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.out = nn.Linear(hidden_size * 2, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, input_len):
        batch_size, seq_len, _ = input.size()
        packed_input = pack_padded_sequence(input, input_len, batch_first=True)
        self.lstm.flatten_parameters()
        output, (hidden_state, cell_state) = self.lstm(packed_input)
        unpacked_output, _ = pad_packed_sequence(output, batch_first=True)
        unpacked_output = unpacked_output.contiguous()
        out = unpacked_output.view(-1, unpacked_output.size(2))
        out = self.out(out)
        out = self.log_softmax(out)
        out = out.contiguous()
        out = out.view(batch_size, seq_len, vocab_size)
        return out

    def create_variable(self, tensor):
        return Variable(tensor.to(device))

    def sent_to_seq(self, input):
        return torch.LongTensor([token_to_ind[w] for w in input])

    def pad_sequence(self, input):
        max_len = len(input[0])
        batch_size = len(input)
        input_seq = torch.zeros(batch_size, max_len).long()
        input_len = list()
        for i, seq in enumerate(input):
            input_len.append(len(seq))
            input_seq[i, :len(seq)] = self.sent_to_seq(seq)
        input_len = torch.LongTensor(input_len)
        return input_seq, input_len

    def create_batch(self, input, output):
        input_seq = [list(w) for w in list(input)]
        target_seq = [list(w) for w in list(output)]

        seq_pairs = sorted(zip(input_seq, target_seq), key=lambda p: len(p[0]), reverse=True)
        input_seq, target_seq = zip(*seq_pairs)

        input_seq, input_len = self.create_batch_tensor(input_seq)
        target_seq, _ = self.create_batch_tensor(target_seq)
        return self.create_variable(input_seq), \
               self.create_variable(input_len), \
               self.create_variable(target_seq)

    def create_batch_tensor(self, input):
        input_seq, input_len = self.pad_sequence(input)
        return input_seq, input_len

    def generate_input(self, input):
        input_seq, input_len = self.pad_sequence([input])
        input_seq, input_len = self.create_variable(input_seq), self.create_variable(input_len)
        embeds = self.embedding(input_seq)
        return embeds, input_len

lm = LanguageModel(HIDDEN_SIZE, OUTPUT_SIZE, EMBED_SIZE, NUM_LAYERS)
print(lm)
if use_cuda:
    lm = lm.to(device)
loss_function = NLLLoss()
optimizer = torch.optim.Adam(lm.parameters(), lr=LEARNING_RATE)

def get_result(input):
    input_seq, input_len = lm.generate_input(input)
    output = lm(input_seq, input_len)
    output_index = torch.max(output, 2)[1]
    output_index = output_index.cpu().numpy().tolist()
    suggest_word = ''.join([ind_to_token[w] for w in output_index[0]])
    return suggest_word

def calculate_perplexity(output_tensor, output_len):
    total_log = 0
    tensor_size = output_tensor.size(0)
    for i, tensor in enumerate(output_tensor):
        tensor = torch.exp(tensor)
        # multiply all value in prob
        p_si = tensor[:output_len[i]]
        p_si = p_si.cpu().detach().numpy()
        p_si = p_si.astype(dtype=float).prod()
        log_si = math.log(p_si, 2)
        total_log += log_si
    total_log = - (total_log/tensor_size)
    return math.pow(2, total_log)


def evaluate():
    max = -1
    for i, (x_data, y_data) in enumerate(test_loader):
        input_seq, input_len, target = lm.create_batch(x_data, y_data)
        embeds = lm.embedding(input_seq)
        output = lm(embeds, input_len)
        a = torch.max(output, 2)
        perplexity = calculate_perplexity(a[0], input_len)
        if perplexity > max:
            max = perplexity
    return max

# evaluate()

def train():
    min_perplexity = 1000
    for epoch in range(NUM_EPOCH):
        print('START EPOCH ----------------%d----------------' % (epoch))
        for i, (x_data, y_data) in enumerate(dataloader):
            input_seq, input_len, target = lm.create_batch(x_data, y_data)
            embeds = lm.embedding(input_seq)

            output = lm(embeds, input_len)
            output = output.view(-1, output.size(2))
            target = target.view(-1)

            loss = loss_function(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print('Interation %d, loss: %.4f , [%d/%d] ' % (i, loss.item(), i * BATCH_SIZE, data_size))
        perplexity = evaluate()
        print('Perplexity : %.4f' % perplexity)
        if perplexity < min_perplexity:
            min_perplexity = perplexity
        torch.save(lm, 'model/language_model.mdl')
        print('__________________Model Saved__________________')

    torch.save(lm, 'model/language_model.mdl')



