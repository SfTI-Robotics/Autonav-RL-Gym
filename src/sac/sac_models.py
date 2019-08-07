import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import numpy as np


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear1.bias.data.uniform_(-init_w, init_w)
        self.linear2.weight.data.uniform_(-init_w, init_w)
        self.linear2.bias.data.uniform_(-init_w, init_w)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


# Critic
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear1.bias.data.uniform_(-init_w, init_w)
        self.linear2.weight.data.uniform_(-init_w, init_w)
        self.linear2.bias.data.uniform_(-init_w, init_w)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-10, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear1.bias.data.uniform_(-init_w, init_w)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear2.weight.data.uniform_(-init_w, init_w)
        self.linear2.bias.data.uniform_(-init_w, init_w)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        # action = z.detach().numpy()

        action = action.detach().numpy()
        return action[0]


class Trainer:
    def __init__(self, state_dim, action_dim, v_min, v_max, w_min, w_max, ram, savePath):

        self.v_min = v_min
        self.v_max = v_max
        self.w_min = w_min
        self.w_max = w_max

        self.value_net = ValueNetwork(state_dim, 128)
        self.target_value_net = ValueNetwork(state_dim, 128)

        self.soft_q_net = SoftQNetwork(state_dim, action_dim, 128)
        self.policy_net = PolicyNetwork(state_dim, action_dim, 128)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()

        value_lr = 3e-4
        soft_q_lr = 3e-4
        policy_lr = 3e-4

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.replay_buffer = ram
        self.savePath = savePath

        print('State Dimensions: ' + str(state_dim))
        print('Action Dimensions: ' + str(action_dim))

    def save_models(self, episode_count):
        torch.save(self.policy_net.state_dict(), self.savePath + '/Models/sac/' + str(episode_count) + '_policy_net.pth')
        torch.save(self.value_net.state_dict(), self.savePath + '/Models/sac/' + str(episode_count) + 'value_net.pth')
        torch.save(self.soft_q_net.state_dict(), self.savePath + '/Models/sac/' + str(episode_count) + 'soft_q_net.pth')
        torch.save(self.target_value_net.state_dict(),
                   self.savePath + '/Models/sac/' + str(episode_count) + 'target_value_net.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load_models(self, episode):
        self.policy_net.load_state_dict(torch.load(self.savePath + '/Models/sac/' + str(episode) + '_policy_net.pth'))
        self.value_net.load_state_dict(torch.load(self.savePath + '/Models/sac/' + str(episode) + 'value_net.pth'))
        self.soft_q_net.load_state_dict(torch.load(self.savePath + '/Models/sac/' + str(episode) + 'soft_q_net.pth'))
        self.target_value_net.load_state_dict(torch.load(self.savePath + '/Models/sac/' + str(episode) + 'target_value_net.pth'))
        print('***Models load***')

    # ----------------------------------------

    def get_action(self, state):

        def action_unnormalized(action, high, low):
            action = low + (action + 1.0) * 0.5 * (high - low)
            action = np.clip(action, low, high)
            return action

        action = self.policy_net.get_action(state)
        return np.array([action_unnormalized(action[0], self.v_max, self.v_min),
                                  action_unnormalized(action[1], self.w_max, self.w_min)])


    def soft_q_update(self, batch_size,
                      gamma=0.99,
                      mean_lambda=1e-3,
                      std_lambda=1e-3,
                      z_lambda=0.0,
                      soft_tau=1e-2,
                      ):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1)
        # print('done', done)

        expected_q_value = self.soft_q_net(state, action)
        expected_value = self.value_net(state)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)

        target_value = self.target_value_net(next_state)
        next_q_value = reward + (1 - done) * gamma * target_value
        q_value_loss = self.soft_q_criterion(expected_q_value, next_q_value.detach())

        expected_new_q_value = self.soft_q_net(state, new_action)
        next_value = expected_new_q_value - log_prob
        value_loss = self.value_criterion(expected_value, next_value.detach())

        log_prob_target = expected_new_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

        mean_loss = mean_lambda * mean.pow(2).mean()
        std_loss = std_lambda * log_std.pow(2).mean()
        z_loss = z_lambda * z.pow(2).sum(1).mean()

        policy_loss += mean_loss + std_loss + z_loss

        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward()
        self.soft_q_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

