import torch
import gym
import numpy as np
from models import LinearActorCriticModel
from agents import CartPoleAgent

hidden_size = 256

env = gym.make("CartPole-v1")
num_inputs = env.observation_space.shape[0]
num_outputs = env.action_space.n

model = LinearActorCriticModel(num_inputs, num_outputs, hidden_size)
model.load_state_dict(torch.load('cartpole-classic.pth', map_location='cpu'))
model.eval()

agent = CartPoleAgent('cartpole-classic', env, model, None, 3000, 500)

agent.play()
