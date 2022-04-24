import torch
from torch import nn

from kronecker_rnn import KfRNNCell

class DocModel(nn.Module):
    def __init__(self, embedding_size, num_heads,
                 attention_dropout, kv_dim, pad_embedding,
                 doc_output_size):
        super(DocModel, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.kv_dim = kv_dim
        self.doc_output_size = doc_output_size

        self.lin_q = nn.Linear(self.embedding_size, kv_dim)
        self.lin_k = nn.Linear(self.embedding_size, kv_dim)
        self.lin_v = nn.Linear(self.embedding_size, kv_dim)

        self.attention = nn.MultiheadAttention(kv_dim, num_heads, dropout=attention_dropout,  # embedding_size
                                               kdim=kv_dim, vdim=kv_dim, batch_first=True)

        self.lin_out = nn.Linear(kv_dim, self.doc_output_size)  # self.embedding_size

        self.pad_embedding = pad_embedding.unsqueeze(0)

    def forward(self, inputs):
        """ inputs of shape (L, E), L - sequence length, E - embedding size """
        assert inputs.shape[1] == self.embedding_size
        inputs = torch.cat([self.pad_embedding, inputs], dim=0).unsqueeze(0)

        q, k, v = self.lin_q(inputs), self.lin_k(inputs), self.lin_v(inputs)
        outputs, weights = self.attention(q, k, v, need_weights=True)
        outputs = outputs[0][0]  # shape = (E,)
        outputs = self.lin_out(outputs)
        return outputs


class KfModel(nn.Module):

    def __init__(self, ts_features, ts_hidden):
        super(KfModel, self).__init__()
        self.kf = KfRNNCell(ts_features, ts_hidden)
        self.seq1 = nn.Sequential(nn.Linear(ts_hidden, ts_hidden), nn.ReLU())

    def forward(self, x):
        return self.seq1(self.kf(x))


class ScoreModel(nn.Module):

    def __init__(self, doc_hidden, ts_hidden, score_linear, output_size):
        super(ScoreModel, self).__init__()
        self.seq = nn.Sequential(nn.Linear(doc_hidden + ts_hidden, score_linear),
                                 nn.ReLU(), nn.Linear(score_linear, output_size))

    def forward(self, ts_x, doc_x):
        cat_x = torch.cat([ts_x.unsqueeze(0).expand(doc_x.shape[0], ts_x.shape[0]), doc_x], axis=1)

        out = self.seq(cat_x)
        return out
