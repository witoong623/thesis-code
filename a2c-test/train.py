import gym
import torch.optim as optim
from agent import CartPoleAgent
from main import ActorCriticModel


hidden_size = 256
learning_rate = 3e-4


if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n

    model = ActorCriticModel(num_inputs, num_outputs, hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    agent = CartPoleAgent('cartpole-classic', env, model, optimizer, 3000, 500)
    agent.train()
