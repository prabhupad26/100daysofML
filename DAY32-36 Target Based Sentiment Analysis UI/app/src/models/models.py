import torch.nn as nn
import torch
from layers import DynamicLSTM, SqueezeEmbedding


class LSTM(nn.Module):
    """ Standard LSTM """

    def __init__(self, embedding_matrix, embed_dim, hidden_dim, polarities_dim):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = DynamicLSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dense = nn.Linear(hidden_dim, polarities_dim)

    def forward(self, inputs):
        text = inputs[0]
        x = self.embed(text)
        x_len = torch.sum(text != 0, dim=-1)
        _, (h_n, _) = self.lstm(x, x_len)
        out = self.dense(h_n[0])
        return out


class AE_LSTM(nn.Module):
    """ LSTM with Aspect Embedding """

    def __init__(self, embedding_matrix, opt):
        super(AE_LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicLSTM(opt.embed_dim * 2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text, aspect_text = inputs[0], inputs[1]
        x_len = torch.sum(text != 0, dim=-1)
        x_len_max = torch.max(x_len)
        aspect_len = torch.sum(aspect_text != 0, dim=-1).float()

        x = self.embed(text)
        x = self.squeeze_embedding(x, x_len)
        aspect = self.embed(aspect_text)
        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.view(aspect_len.size(0), 1))
        aspect = torch.unsqueeze(aspect_pool, dim=1).expand(-1, x_len_max, -1)
        x = torch.cat((aspect, x), dim=-1)

        _, (h_n, _) = self.lstm(x, x_len)
        out = self.dense(h_n[0])
        return out