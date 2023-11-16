import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import *
from utils import get_sequence_mask


class Generator(nn.Module):
    def __init__(self, hidden_dim=32, emb_dim=32, kc_num=123, exercise_num=17714, user_num=4163, device='cpu') -> None:
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.exercise_num = exercise_num
        self.kc_num = kc_num
        self.device = device

        ########################### Encoder To Extract Student Personal Info ###############################
        # Pred Module
        self.pred = nn.Linear(128, 1)
        ####################################################################################################

        # Student Tracing
        self.actor_rnn = nn.GRU(input_size=self.emb_dim * 2, hidden_size=self.emb_dim, batch_first=True)

        # Action Module
        self.action = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(3 * emb_dim, 1),
            nn.Sigmoid(),
        )

        # Value Module
        self.critic = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(3 * emb_dim, 1),
            nn.LeakyReLU(),
        )

        # kc2hidden
        self.kc = nn.Linear(self.kc_num, self.emb_dim)

        # exer2emb
        self.exercise_emb = nn.Embedding(exercise_num + 1, self.emb_dim, padding_idx=exercise_num)

        # user2emb
        self.user_emb = nn.Linear(self.kc_num, self.emb_dim)

        # action2emb
        self.action_emb = nn.Embedding(3, self.emb_dim, padding_idx=2)

    def init_model(self):
        # init_model
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, user, questions, tags, actor_rnn_state=None, answers=None):
        kc_emb = self.kc(tags)
        exer_emb = self.exercise_emb(questions)
        batch_size = questions.shape[0]
        seq_len = questions.shape[1]
        if actor_rnn_state is None:
            actor_rnn_state = Variable(torch.Tensor(batch_size, self.emb_dim).uniform_(0, 1))
            actor_rnn_state = actor_rnn_state.to(self.device)
        else:
            actor_rnn_state = self.user_emb(actor_rnn_state)
        return_logits = []
        return_value = []
        return_answer = []

        for i in range(seq_len):
            pred = self.action(
                torch.cat([actor_rnn_state, kc_emb[:, i], exer_emb[:, i]], dim=-1)
            )  # batch_size * 1
            answer = torch.where(pred >= 0.5, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device)).to(
                self.device)  # batch_size * 1
            value = self.critic(
                torch.cat([actor_rnn_state, kc_emb[:, i], exer_emb[:, i]], dim=-1)
            )  # batch_size * 1

            return_logits.append(pred)
            return_value.append(value)
            return_answer.append(answer)

            if answers is None:
                ans = pred.view(-1)
            else:
                ans = answers[:, i]

            gru_input = torch.cat(
                [
                    kc_emb[:, i] * (ans.view(-1, 1) >= 0.5).type_as(kc_emb[:, i]).expand_as(kc_emb[:, i]),
                    kc_emb[:, i] * (ans.view(-1, 1) < 0.5).type_as(kc_emb[:, i]).expand_as(kc_emb[:, i])
                ], dim=-1
            ).unsqueeze(1)  # batch_size * 1 * 1

            actor_rnn_state = actor_rnn_state.unsqueeze(0)  # batch_size * 1 * 1
            _, actor_rnn_state = self.actor_rnn(gru_input, actor_rnn_state)
            actor_rnn_state = actor_rnn_state.squeeze(0)
        return torch.stack(return_logits, dim=1), torch.stack(return_value, dim=1), torch.stack(return_answer, dim=1)
