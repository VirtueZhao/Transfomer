import math

import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import torch
import torch.nn as nn
from torch.autograd import Variable


class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        """
        vocab: size of vocabulary
        d_model: dimension of work embedding
        """
        super(Embeddings, self).__init__()

        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        # Positional Embedding Matrix
        pe = torch.zeros(max_len, d_model)
        # Absolute Position Matrix
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)

        return self.dropout(x)


# d_model = 512
# vocab = 1000

# x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))

# emb = Embeddings(vocab, d_model)
# embr = emb(x)
# print(embr)
# print(x.shape)
# print(embr.shape)

# dropout = 0.1
# max_len = 60

# pe = PositionalEncoding(d_model, dropout, max_len)
# pe_result = pe(embr)
# print(pe_result)
# print(pe_result.shape)

# Visualize Positional Embedding
# plt.figure(figsize=(15, 5))
# pe = PositionalEncoding(20, 0)
# y = pe(Variable(torch.zeros(1, 100, 20)))
# print(y.shape)
# plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
# plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
# plt.savefig("visualization_positional_embedding.png")
