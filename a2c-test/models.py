import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU


class LinearActorCriticModel(nn.Module):
    def __init__(self, num_inputs, num_actions,
                   hidden_size, learning_rate=3e-4):
        super(LinearActorCriticModel, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)

        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist


def create_2d_conv_rom(input_channel):
    ''' Rom version that confirm to be working '''
    return nn.Sequential(
        nn.Conv2d(input_channel, 128, 5, stride=2),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, 5, stride=2),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 32, 5, stride=2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
    )


def create_2d_conv(input_channel):
    return nn.Sequential(
        nn.Conv2d(input_channel, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
    )


class CNNActorCriticModel(nn.Module):
    def __init__(self, input_channel, num_actions, hidden_size):
        super().__init__()
        
        self.policy_conv = create_2d_conv(input_channel)
        self.policy_linear = nn.Sequential(
            nn.Linear(8064, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
            nn.Softmax(dim=0)
        )

        self.value_cov = create_2d_conv(input_channel)
        self.value_linear = nn.Sequential(
            nn.Linear(8064, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        ''' state is image tensor '''
        value_x = self.value_cov(state)
        value_x = value_x.view(-1)
        value = self.value_linear(value_x)

        policy_x = self.policy_conv(state)
        policy_x = policy_x.view(-1)
        policy = self.policy_linear(policy_x)

        return value, policy
