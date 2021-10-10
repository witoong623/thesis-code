import gym
from torch._C import device
import torch.nn as nn
import torch.optim as optim
from agents import DDQNCNNAgent
from models import CNNQNetwork


learning_rate = 0.00005
hidden_size = 512
num_episodes = 3000
num_images = 4
num_episodes = 3000

# run with command: xvfb-run -s "-screen 0 1400x900x24" python train_dqn_cartpole.py
if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n

    loss_func = nn.MSELoss()

    next_q_model = CNNQNetwork(num_images, num_actions, hidden_size)
    eval_q_model = CNNQNetwork(num_images, num_actions, hidden_size)

    optimizer = optim.Adam(eval_q_model.parameters(), lr=learning_rate)

    agent = DDQNCNNAgent(
        'DDQN-CNN-cartpole', env, next_q_model, eval_q_model, num_actions, loss_func,
        optimizer, num_episodes, (200, 80), num_images, crop_top=160, replace_next=250,
        eps_dec=4e-5, device='cuda:0', batch_size=64
    )

    agent.train()
