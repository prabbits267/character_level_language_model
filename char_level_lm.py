import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
BATCH_SIZE = 64


from my_dataset import *
dataset = MyDataset()
dataloader = DataLoader(dataset=dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True)

full_text = dataset.text

tokens = sorted(set(full_text))

token_to_ind = {'<BEGIN>':1, '<END>':2, '<PAD>':0}
ind_to_token = {1:'<BEGIN>', 2:'<END>', 0:'<PAD>'}

token_to_ind = {w:i for i,w in enumerate(tokens)}
ind_to_token = {token_to_ind[w]:w for w in token_to_ind}
vocab_size = len(tokens)
use_cuda = torch.cuda.is_available()
device = ('cuda:0' if use_cuda else 'cpu')

# class LanguageModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, embed_size, num_layers):
#         super(LanguageModel, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_size)
#         self.lstm = nn.LSTM(
#             input_size = embed_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             output_size=vocab_size,
#             batch_first=True
#         )
#         self.softmax = nn.Softmax()
#
#     def forward(self, input, input_len):
#         batch_size, seq_len = input.size()
#         packed_input = pack_padded_sequence(input, input_len)
#         output, hidden_state, cell_state = self.lstm(packed_input)
#         unpacked_output, _ = pad_packed_sequence(output)
#         out = unpacked_output.view(-1, unpacked_output.size(2))
#         out = self.softmax(out)
#         out = out.view(batch_size, seq_len, vocab_size)
#         return out
#
#     def create_variable(self,tensor):
#         return Variable(tensor.to(device))



for i, (x_data, y_data) in enumerate(dataloader):
    print(x_data)
    print(y_data)

