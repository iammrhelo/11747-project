import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x, h):
        embed_x = self.embed(x)

        output, h = self.lstm(embed_x)

        return output, h


if __name__ == "__main__":

    vocab_size = 5
    embed_size = 10
    hidden_size = 5
    num_layers = 1

    model = RNNLM(vocab_size, embed_size, hidden_size, num_layers)

    # Dummy data
    inputs = torch.IntTensor(np.array(range(5)))
    state0 = torch.FloatTensor(np.zeros(5))
    targets = torch.FloatTensor(np.array([1,0,0,0,0]))

    input_tensor = Variable(inputs)
    state_tensor = Variable(state0)
    target_tensor = Variable(targets)

    import pdb; pdb.set_trace()
