import gym
import torch.optim as optim
from agents import CartPole2DAgent
from models import CNNActorCriticModel


hidden_size = 256
learning_rate = 0.000005
num_images = 2

# run with command: xvfb-run -s "-screen 0 1400x900x24" python train_cnn_cartpole.py
if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    num_actions = env.action_space.n

    model = CNNActorCriticModel(num_images, num_actions, hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    agent = CartPole2DAgent('cartpole-2d', env, model, 3000, 500, num_images, (200, 80), skip_frame=1, temperature=0.003, crop_top=160, device='cuda:0')
    agent.train(optimizer, True)
