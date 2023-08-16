import torch
from utils import sequence_mask


def GAE(reward, value, gamma, lam, origin_len, device):
    if reward.shape == value.shape:
        ppo_adv = torch.zeros_like(reward)
        ppo_returns = torch.zeros_like(reward)
        for i, length in enumerate(origin_len):
            adv = torch.zeros(size=(length,), dtype=torch.float).to(device)
            delta = torch.zeros(size=(length,), dtype=torch.float).to(device)
            pre_value, pre_adv = 0, 0
            for j in reversed(range(length)):
                delta[j] = reward[i, j, 0] + gamma * pre_value - value[i, j, 0]
                adv[j] = delta[j] + gamma * lam * pre_adv
                pre_adv = adv[j]
                pre_value = value[i, j, 0]
            ppo_returns[i,:length] = value[i, :length] + adv.reshape(-1, 1)
            adv = (adv - torch.mean(adv)) / (torch.std(adv, unbiased=False) + 1e-5)
            ppo_adv[i, :length] = adv.reshape(-1 ,1)
        return ppo_adv, ppo_returns
    else:
        ppo_adv = torch.zeros_like(value)
        ppo_returns = torch.zeros_like(value)
        for i, length in enumerate(origin_len):
            adv = torch.zeros(size=(length,), dtype=torch.float).to(device)
            delta = torch.zeros(size=(length,), dtype=torch.float).to(device)
            pre_value, pre_adv = 0, 0
            for j in reversed(range(length)):
                if j == length - 1:
                    delta[j] = reward[i, 0] + gamma * pre_value - value[i, j, 0]
                    adv[j] = delta[j] + gamma * lam * pre_adv
                    pre_adv = adv[j]
                    pre_value = value[i, j, 0]
                else:
                    delta[j] = 0 + gamma * pre_value - value[i, j, 0]
                    adv[j] = delta[j] + gamma * lam * pre_adv
                    pre_adv = adv[j]
                    pre_value = value[i, j, 0]
            ppo_returns[i,:length] = value[i, :length] + adv.reshape(-1, 1)
            adv = (adv - torch.mean(adv)) / (torch.std(adv, unbiased=False) + 1e-5)
            ppo_adv[i, :length] = adv.reshape(-1 ,1)
        return ppo_adv, ppo_returns


def PPO_STEP(G, action, advantages, returns, origin_len, fixed_log_prob, optim_g, user_feature, question, tag, args):
    G.train()
    logits, value, _ = G.forward(None, question, tag, actor_rnn_state=user_feature, answers=None)
    v_loss = torch.mean(sequence_mask((value - returns), origin_len).pow(2))
    dist = torch.distributions.Bernoulli(logits)
    log_prob = dist.log_prob(action.float())
    ratio = torch.exp(log_prob - fixed_log_prob)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages
    p_loss = torch.sum(sequence_mask(-torch.minimum(surr1, surr2), origin_len)) / sum(origin_len)

    # 计算熵值，控制动作分布，避免全0全1出现
    entropy = 0
    for i in range(args.generate_size):
        entropy += torch.distributions.Bernoulli(logits[i, :origin_len[i]]).entropy().mean()
    entropy /= args.generate_size
    
    loss = v_loss + p_loss - 0.01 * entropy
    optim_g.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(G.parameters(), 10)
    optim_g.step()
    return v_loss.item(), p_loss.item(), entropy.item(), loss.item()