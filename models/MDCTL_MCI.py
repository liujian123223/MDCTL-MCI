import torch.nn as nn
import pytorch_lightning as pl

class ICB(pl.LightningModule):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1 = self.act(x1)
        x1 = self.drop(x1)
        x2 = self.conv2(x)
        x2 = self.act(x2)
        x2 = self.drop(x2)
        out = x1 * x2
        out = self.conv3(out)
        out = out.transpose(1, 2)
        out += residual
        return out

class ResBlock(nn.Module):
    def __init__(self, configs):
        super(ResBlock, self).__init__()
        self.temporal = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.seq_len),
            nn.Dropout(configs.dropout)
        )
        self.channel = nn.Sequential(
            nn.Linear(configs.enc_in * (configs.subsequence_num + 1), configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.enc_in * (configs.subsequence_num + 1)),
            nn.Dropout(configs.dropout)
        )
        self.ICB = ICB(in_features=configs.enc_in * (configs.subsequence_num + 1), hidden_features=configs.ICB_hidden, drop=configs.dropout)


    def forward(self, x):
        res =x
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)
        x = x + self.ICB(x)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.layer = configs.e_layers
        self.model = nn.ModuleList([ResBlock(configs) for _ in range(configs.e_layers)])
        self.pred_len = configs.pred_len
        self.projection = nn.Linear(configs.seq_len, configs.pred_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        enc_out = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)
        return enc_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        else:
            raise ValueError('Only forecast tasks implemented yet')