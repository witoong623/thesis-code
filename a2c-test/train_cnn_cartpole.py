import gym
import torch.optim as optim
from agents import CartPole2DAgent
from models import CNNActorCriticModel


hidden_size = 256
learning_rate = 0.0007
num_images = 4


if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n

    model = CNNActorCriticModel(num_images, num_outputs, hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    agent = CartPole2DAgent('cartpole-2d', env, model, 3000, 500, num_images, (200, 133), device='cuda:0')
    agent.train(optimizer, True)