import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class Disc(nn.Module):
    def __init__(self, n_output=1, hidden_dim=128, emb_dim=128, exercise_num=17714, kc_num=123, device='cpu'):
        super(Disc, self).__init__()
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.exercise_num = exercise_num
        self.kc_num = kc_num
        self.output_dim = n_output
        self.device = device

        # 定义模型结构
        self.user_emb = nn.Linear(self.kc_num, self.emb_dim)
        self.kc = nn.Linear(self.kc_num, self.emb_dim)
        self.ex_emb = nn.Embedding(self.exercise_num + 1, self.emb_dim, padding_idx=self.exercise_num)
        self.rnn = nn.GRU(input_size=self.emb_dim * 4, hidden_size=self.emb_dim, batch_first=True,
                          bidirectional=False)
        self.dropout_linear = nn.Dropout(p=0.5)
        self.hidden2out = nn.Linear(self.emb_dim, self.output_dim)
        self.init_model()
        print("Disc Init Done!")

    def init_model(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
                # print(name, param)

    # use gru output to predict reward
    def forward(self, traj):
        feat = traj[0] # batch_size * seq_len * kc_num
        question = traj[1] # batch_size * seq_len * 1
        tag = traj[2] # batch_size * seq_len * kc_num
        answer = traj[3] # batch_size * seq_len * 1
        feat_emb = self.user_emb(feat)
        exer_emb = self.ex_emb(question)
        tag_emb = self.kc(tag)
        gru_input = torch.cat(
            [
                exer_emb,
                tag_emb
            ],
            dim=-1
        )
        gru_input = torch.cat(
            [
                gru_input * (answer >= 0.5).type_as(gru_input).expand_as(gru_input),
                gru_input * (answer < 0.5).type_as(gru_input).expand_as(gru_input)
            ],
            dim=-1
        )

        feat_emb = feat_emb.unsqueeze(0)
        out, _ = self.rnn(gru_input, feat_emb)
        out = self.dropout_linear(out)
        reward = self.hidden2out(out)
        return reward
