import data
import random
from sklearn.metrics import roc_auc_score
from rouge import Rouge
from network import *
from utils import *
from torch.utils.data import DataLoader
import numpy as np
import torch
import logging
from network import *
from ppo import GAE, PPO_STEP
from tqdm import tqdm
from utils import sequence_mask, get_str_for_eval
import os
import torch.nn.functional as F


class MAILModel:
    def __init__(self, args) -> None:
        self.device = "cuda:" + str(args.cuda) if torch.cuda.is_available() else "cpu"
        self.batch_size = args.batch_size
        self.args = args
        self.G = Generator(hidden_dim=args.hidden_size, emb_dim=args.embed_size, kc_num=args.kc_num,
                               exercise_num=args.exercise_num, user_num=args.user_num, device=self.device).to(
                self.device)
        self.D = Disc(n_output=1, hidden_dim=args.hidden_size, emb_dim=args.embed_size, kc_num=args.kc_num,
                      exercise_num=args.exercise_num, device=self.device).to(self.device)

    def eval(self, g, val_set):
        g.eval()
        eval_loader = DataLoader(
            val_set,
            shuffle=False,
            num_workers=4,
            batch_size=32,
            drop_last=True,
            collate_fn=self.dealing
        )
        target = []
        pred_1 = []
        target_auc = []
        pred_2 = []
        for i, (early, late) in tqdm(enumerate(eval_loader)):
            question = late[1].to(self.device)
            answer = late[2].to(self.device)
            tag = late[3].to(self.device)
            origin_len = late[4]
            user_feature = self.extract(early, mode='easy')
            _, _, pred = g.forward(None, question, tag, actor_rnn_state=user_feature, answers=None)
            target.extend(get_str_for_eval(questions=question.cpu().tolist(), answers=answer.cpu().tolist()))
            pred_1.extend(
                get_str_for_eval(questions=question.cpu().tolist(), answers=pred.reshape_as(answer).cpu().tolist()))
            target_auc.extend(answer.cpu().tolist())
            pred_2.extend(pred.reshape_as(answer).cpu().tolist())
        rouger = Rouge()

        rouge = rouger.get_scores(pred_1, target, avg=True)
        auc = roc_auc_score(target_auc, pred_2)

        return rouge, auc

    def extract(self, early, mode):
        if mode == 'easy':
            tag = early[3].to(self.device)
            answer = early[2].to(self.device)
            origin_len = early[4]
            feat = []
            for i, length in enumerate(origin_len):
                user_feature = tag[i, :length] * answer[i, :length].unsqueeze(-1)
                feat_i = torch.mean(user_feature, dim=0)
                feat.append(feat_i)
            feat = torch.stack(feat, dim=0).to(self.device)
        else:
            # Use Encoder to extract user feature
            pass
        return feat

    def train(self, args, train_set, val_set, test_set):
        print(self.G)
        best_rouge = 0

        if args.optim == 'adam':
            optim_g = torch.optim.Adam(self.G.parameters(), lr=args.g_lr)
            optim_d = torch.optim.Adam(self.D.parameters(), lr=args.d_lr)
            optim_g_temp = torch.optim.Adam(self.G.parameters(), lr=args.g_lr * args.gailoss)
        else:
            optim_g = torch.optim.SGD(self.G.parameters(), lr=args.g_lr)
            optim_d = torch.optim.SGD(self.D.parameters(), lr=args.d_lr)

        d_loss_func = nn.BCELoss(reduction='none')

        train_loader = DataLoader(
            train_set,
            shuffle=True,
            num_workers=4,
            batch_size=args.generate_size,
            drop_last=True,
            collate_fn=self.dealing
        )

        if not os.path.exists(f'param/{args.log_name}'):
            os.mkdir(f'param/{args.log_name}')

        step = 0
        ppo_step = 0
        for ep in range(args.epoch):
            for early, late in train_loader:
                step += 1
                question = late[1].to(self.device)
                answer = late[2].to(self.device)
                tag = late[3].to(self.device)
                origin_len = np.array(late[4])
                answer = answer.unsqueeze(-1)
                user_feature = self.extract(early, mode='easy')
                similarity_matrix = F.cosine_similarity(user_feature.unsqueeze(1),
                                                        user_feature.unsqueeze(0), dim=2, eps=1e-30)
                _, indices = torch.topk(input=similarity_matrix, k=2, dim=1)
                temp_user_feature = user_feature[indices[:, 1].view(-1)]

                self.G.eval()
                self.D.train()

                with torch.no_grad():
                    _, _, action_sample = self.G.forward(None, question, tag,
                                                         actor_rnn_state=temp_user_feature,
                                                         answers=None)

                real_traj = [user_feature, question, tag, answer]
                fake_traj = [temp_user_feature, question, tag, action_sample]
                real_out = self.D.forward(real_traj)
                gen_out = self.D.forward(fake_traj)

                ones = torch.ones_like(real_out)
                zeros = torch.zeros_like(gen_out)
                real_loss = d_loss_func(torch.sigmoid(real_out), ones)
                gen_loss = d_loss_func(torch.sigmoid(gen_out), zeros)
                adversarial_loss = 0.5 * real_loss + 0.5 * gen_loss
                adversarial_loss = torch.mean(adversarial_loss)

                if args.d_sample:
                    pairwise_loss = 0
                    sample_time = 3
                    for _ in range(sample_time):
                        pairwise_loss += torch.mean(-torch.log(torch.sigmoid(real_out - gen_out)))
                        random_index = torch.randperm(gen_out.shape[0])
                        gen_out = gen_out[random_index]
                    pairwise_loss = pairwise_loss / sample_time
                else:
                    mean_real = torch.mean(real_out)
                    mean_gen = torch.mean(gen_out)
                    pairwise_loss = torch.mean(-torch.log(1e-9 + torch.sigmoid(real_out - mean_gen))) + torch.mean(
                        torch.log(1e-9 + torch.sigmoid(gen_out - mean_real)))

                dloss = args.tau * pairwise_loss + (1 - args.tau) * adversarial_loss
                optim_d.zero_grad()
                dloss.backward()
                optim_d.step()

                self.D.eval()
                self.G.train()

                with torch.no_grad():
                    if args.no_di is False:
                        logits_expert, value_expert, action_expert = self.G.forward(None, question, tag,
                                                                                    actor_rnn_state=user_feature,
                                                                                    answers=None)

                        dist = torch.distributions.Bernoulli(logits_expert)
                        fixed_log_prob = dist.log_prob(action_expert.float())
                        reward = torch.where(action_expert == answer, torch.tensor(1.).to(self.device),
                                             torch.tensor(0.).to(self.device))
                        advantages, returns = GAE(reward, value_expert, gamma=args.gamma, lam=args.lam,
                                                  origin_len=origin_len, device=self.device)

                    if args.no_gail is False:
                        logits_temp, value_temp, action_temp = self.G.forward(None, question, tag,
                                                                              actor_rnn_state=temp_user_feature,
                                                                              answers=None)
                        dist_temp = torch.distributions.Bernoulli(logits_temp)
                        temp_log_prob = dist_temp.log_prob(action_temp.float())
                        if args.g_mean:
                            real_mean_reward = torch.mean(self.D.forward([user_feature, question, tag, answer], origin_len))
                            reward_temp = torch.sigmoid(
                                self.D.forward([temp_user_feature, question, tag, action_temp], origin_len) - real_mean_reward
                            )
                        else:
                            reward_temp = torch.sigmoid(self.D.forward([temp_user_feature, question, tag, action_temp], origin_len))

                        logging.info(f'Step: {step},  temp reward: {torch.mean(reward_temp).item()}')
                        advantages_temp, returns_temp = GAE(reward_temp, value_temp, gamma=args.gamma, lam=args.lam,
                                                            origin_len=origin_len, device=self.device)

                for pep in range(args.ppo_epoch):
                    ppo_step += 1
                    if args.no_di is False:
                        v_loss, p_loss, entropy, total_loss = PPO_STEP(
                            G=self.G,
                            action=action_expert,
                            advantages=advantages,
                            returns=returns,
                            origin_len=origin_len,
                            fixed_log_prob=fixed_log_prob,
                            optim_g=optim_g,
                            user_feature=user_feature,
                            question=question,
                            tag=tag,
                            args=args
                        )

                    if args.no_gail is False:
                        temp_v_loss, temp_p_loss, temp_entropy, temp_total_loss = PPO_STEP(
                            G=self.G,
                            action=action_temp,
                            advantages=advantages_temp,
                            returns=returns_temp,
                            origin_len=origin_len,
                            fixed_log_prob=temp_log_prob,
                            optim_g=optim_g_temp,
                            user_feature=temp_user_feature,
                            question=question,
                            tag=tag,
                            args=args
                        )

            valid_rouge, _ = self.eval(g=self.G, val_set=val_set)
            test_rouge, _ = self.eval(g=self.G, val_set=test_set)

            if valid_rouge['rouge-l']['f'] > best_rouge:
                self.save_model(model=self.G, model_name=f'best_train_g', experiment_name=args.log_name)
                self.save_model(model=self.D, model_name=f'best_train_d', experiment_name=args.log_name)
            if ep % 10 == 0:
                self.save_model(model=self.G, model_name=f'{ep}_train_g', experiment_name=args.log_name)
                self.save_model(model=self.D, model_name=f'{ep}_train_d', experiment_name=args.log_name)

    def to_onehot(self, tags):
        if tags == '':
            return [0] * self.args.kc_num
        tags = tags.split(';')
        onehot = [0] * self.args.kc_num
        for i in tags:
            onehot[int(i)] = 1
        return onehot

    def dealing(self, traj):
        early, late = zip(*traj)
        return self.padding_single(list(early)), self.dealing_single(list(late))

    def dealing_single(self, traj):
        origin_len = [i['len'] for i in traj]
        user = np.array(
            [i['user_id'] for i in traj]
        )

        padding_question = np.array(
            [i['question_list'] for i in traj])
        padding_answer = np.array(
            [i['answer_list'] for i in traj])

        padding_tags = []
        for i in traj:
            padding_tags.append([self.to_onehot(
                j) for j in i['question_tag_list']
            ])
        padding_tags = np.array(padding_tags)

        user = torch.LongTensor(user)
        padding_question = torch.LongTensor(padding_question)
        padding_answer = torch.LongTensor(padding_answer)
        padding_tags = torch.FloatTensor(padding_tags)

        return user, padding_question, padding_answer, padding_tags, origin_len

    def padding_single(self, traj):
        traj.sort(key=lambda elem: -elem['len'])
        max_len = max(i['len'] for i in traj)
        origin_len = [i['len'] for i in traj]

        user = np.array(
            [i['user_id'] for i in traj]
        )

        padding_question = np.array(
            [i['question_list'] + (max_len - i['len']) * [self.args.exercise_num] for i in traj])
        padding_answer = np.array(
            [i['answer_list'] + (max_len - i['len']) * [0] for i in traj])

        padding_tags = []
        for i in traj:
            padding_tags.append([self.to_onehot(
                j) for j in i['question_tag_list'] + (max_len - i['len']) * ['-1']])
        padding_tags = np.array(padding_tags)

        user = torch.LongTensor(user)
        padding_question = torch.LongTensor(padding_question)
        padding_answer = torch.LongTensor(padding_answer)
        padding_tags = torch.FloatTensor(padding_tags)

        return user, padding_question, padding_answer, padding_tags, origin_len

    def compute_loglikelihood(self, pred_scores, target_scores):
        """
        Compute Loglikelihood
        """
        total_loss = 0.
        cnt = 0
        for idx, score in enumerate(target_scores):
            cnt += 1
            loss = 0.
            if score == 0:
                loss -= torch.log(1 - pred_scores[idx].view(1) + 1e-30)
            else:
                loss -= torch.log(pred_scores[idx].view(1) + 1e-30)
            total_loss += loss
        total_loss /= cnt
        return total_loss.item()

    def save_model(self, model, model_name, experiment_name):
        torch.save(model.state_dict(), f'param/{experiment_name}/{model_name}.pt')
