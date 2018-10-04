from char_level_lm import *


BATCH_SIZE = 64
EMBED_SIZE = 200
HIDDEN_SIZE = 64
OUTPUT_SIZE = vocab_size
NUM_LAYERS = 2


class SuggestText:
    def __init__(self, hidden_size, output_size, embed_size, num_layers):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.lm = LanguageModel(self.hidden_size, self.output_size, self.embed_size, self.num_layers)
        self.path = 'model/language_model.mdl'
        self.lm = torch.load(self.path)

    def suggest(self, text):
        input_seq, input_len = self.lm.generate_input(text)
        output = self.lm(input_seq, input_len)
        max_tensor = torch.max(output, 2)
        token_ind = max_tensor[1].numpy().tolist()
        suggest_word = [ind_to_token[w] for w in token_ind]
        return suggest_word


suggest_word = SuggestText(HIDDEN_SIZE, OUTPUT_SIZE, EMBED_SIZE, NUM_LAYERS)
print(suggest_word.suggest('フレーム着火対応'))








