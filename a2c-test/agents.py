import cv2
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import trange
from torch.distributions import Categorical, Distribution

np.seterr(divide='raise')


class ActorCriticAgent:
    ''' env should be gym environment '''
    def __init__(self, name, env, model, num_episode, max_step, gamma=0.99, temperature=0.001, device='cpu'):
        self.name = name
        self.env = env
        self.model = model

        self.num_episode = num_episode
        self.max_step = max_step
        self.gamma = gamma
        self.temperature = temperature
        self.device = device

        self.all_rewards = []
        self.all_lens = []

        self.model.to(self.device)

    def get_distribution(self, policies_probs) -> Distribution:
        ''' should return class of torch.distributions '''
        raise NotImplementedError()

    def step(self, action):
        ''' step through env, return new state, reward, done '''
        raise NotImplementedError()


    def reset(self):
        ''' should return new state after reset '''
        raise NotImplementedError()


    def _save_training_graph(self):
        Path('graph').mkdir(exist_ok=True)

        smoothed_rewards = pd.Series.rolling(pd.Series(self.all_rewards), 10).mean()
        smoothed_rewards = [elem for elem in smoothed_rewards]
        plt.plot(self.all_rewards)
        plt.plot(smoothed_rewards)
        plt.plot()
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward graph')
        plt.savefig(os.path.join('graph', f'{self.name}.jpg'))

    def train(self, optimizer, save_training_graph=False):
        self.model.train()
        self.all_rewards = []
        self.all_lens = []

        for episode in (tq := trange(self.num_episode)):
            entropies = []
            rewards = []
            log_action_probs = []
            values = []
            optimizer.zero_grad()
            state = self.reset()

            for step in range(self.max_step):
                value, policy_probs = self.model(state)
                action_dist = self.get_distribution(policy_probs)
                action = action_dist.sample()

                state, reward, done = self.step(action.cpu().detach().item())

                rewards.append(reward)
                values.append(value)

                log_action_probs.append(action_dist.log_prob(action))
                # entropy is -np.sum(np.mean(dist) * np.log(dist))
                entropies.append(action_dist.entropy())

                if done or step == self.max_step - 1:
                    Q_value = value.cpu().detach().item()
                    self.all_rewards.append(sum(rewards))
                    self.all_lens.append(step)

                    if episode % 10 == 0:
                        tq.set_description(f'Episode: {episode}, reward: {sum(rewards)}, total length: {step}')

                    break

            Q_values = np.zeros_like(values, dtype='float32')
            for t in reversed(range(len(rewards))):
                Q_value = rewards[t] + self.gamma * Q_value
                Q_values[t] = Q_value

            values = torch.tensor(values, dtype=torch.float32, device=self.device)
            Q_values = torch.tensor(Q_values, dtype=torch.float32, device=self.device)
            log_action_probs = torch.stack(log_action_probs)
            entropies = torch.stack(entropies)

            advantage = Q_values - values
            actor_loss = (-log_action_probs * advantage).mean()
            critic_loss = 0.5 * advantage.pow(2).mean()
            entropy_loss = entropies.sum()
            ac_loss = actor_loss + critic_loss + self.temperature * entropy_loss

            ac_loss.backward()
            optimizer.step()

        torch.save(self.model.state_dict(), f'{self.name}.pth')
        if save_training_graph:
            self._save_training_graph()

    def play(self, render=True):
        self.model.eval()

        with torch.no_grad():
            rewards = 0
            state = self.reset()
            for step in (t := trange(self.max_step)):
                if render:
                    self.env.render()

                _, policy_probs = self.model(state)
                action_dist = self.get_distribution(policy_probs)
                action = action_dist.sample()

                state, reward, done = self.step(action.cpu().detach().item())
                rewards += reward

                t.set_description(f'reward: {rewards}, step: {step}')

                if done:
                    break


class CartPoleAgent(ActorCriticAgent):
    def get_distribution(self, policies_probs):
        return Categorical(probs=policies_probs)

    def step(self, action):
        new_state, reward, done, _ = self.env.step(action)

        return new_state, reward, done

    def reset(self):
        state = self.env.reset()

        return state


class CartPole2DAgent(ActorCriticAgent):
    def __init__(self, name, env, model, num_episode, max_step, num_images, image_size, skip_frame=1, crop_top=0, **kwargs):
        super().__init__(name, env, model, num_episode, max_step, **kwargs)

        self.crop_top = crop_top
        self.num_images = num_images
        self.image_size = image_size
        self.skip_frame = skip_frame
        # this buffer size holds exactly the same amount of buffer needed
        self.image_buffer = np.zeros((self.num_images * self.skip_frame - (self.skip_frame-1), self.image_size[1], self.image_size[0]))
        self.init_buffer = False
        self.frame_count = 0

    def get_distribution(self, policies_probs):
        return Categorical(probs=policies_probs)

    def _save_image(self, image):
        cv2.imwrite(f'output_images/frame_{self.frame_count}.jpg', image)
        self.frame_count += 1

    def _get_image_tensor(self):
        ''' get image, preprocess and return image tensor to be used as state
            credit to: https://pylessons.com/CartPole-PER-CNN/
        '''
        img = self.env.render(mode='rgb_array')
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_rgb = img_rgb[self.crop_top:,:]
        img_rgb_resized = cv2.resize(img_rgb, self.image_size, interpolation=cv2.INTER_CUBIC)
        img_rgb_resized[img_rgb_resized < 255] = 0
        # self._save_image(img_rgb_resized)
        img_rgb_resized = img_rgb_resized / 255

        if not self.init_buffer:
            for t in range(self.image_buffer.shape[0]):
                self.image_buffer[t,:,:] = img_rgb_resized
            self.init_buffer = True

        self.image_buffer = np.roll(self.image_buffer, 1, axis=0)
        self.image_buffer[0,:,:] = img_rgb_resized

        state_images = np.zeros((self.num_images, self.image_size[1], self.image_size[0]))
        for i, t in enumerate(range(0, self.num_images * self.skip_frame, self.skip_frame)):
            state_images[i,:,:] = self.image_buffer[t]
        return torch.tensor(state_images, dtype=torch.float32, device=self.device).unsqueeze(0)

    def step(self, action):
        _, reward, done, _ = self.env.step(action)
        new_state = self._get_image_tensor()

        return new_state, reward, done

    def reset(self):
        self.env.reset()

        state = self._get_image_tensor()

        return state
