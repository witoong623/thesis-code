import cv2
import os
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch._C import device
from tqdm import trange
from torch.distributions import Categorical, Distribution
from replay_buffer import ReplayBuffer

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


class DDQNCNNAgent:
    def __init__(self, name, env, next_q_model, eval_q_model, n_actions,
                 loss_func, optimizer, episode, image_size, num_images,
                 skip_frame=1, crop_top=0, epsilon=0.1, gamma=1, buffer_size=100, 
                 eps_min=0.01, eps_dec=5e-7, batch_size=4, replace_next=1000, device='cpu'):
        self.env = env
        self.name = name

        self.next_q = next_q_model.to(device)
        self.eval_q = eval_q_model.to(device)

        self.gamma = gamma
        self.epsilon = epsilon
        self.actions_space = list(range(n_actions))

        self.loss_func = loss_func
        self.optimizer = optimizer

        self.batch_size = batch_size

        self.episode = episode
        self.learn_counter = 0
        self.replace_next = replace_next
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.device = device
        self.image_size = image_size
        assert skip_frame > 0
        self.skip_frame = skip_frame
        assert crop_top >= 0
        self.crop_top = crop_top
        assert num_images > 0
        self.num_images = num_images

        self.image_buffer = np.zeros((self.num_images * self.skip_frame - (self.skip_frame-1), self.image_size[1], self.image_size[0]))
        self.init_buffer = False
        self.buffer = ReplayBuffer(buffer_size, (num_images, self.image_size[1], self.image_size[0]))
        self.all_rewards = []

    def store_states_action(self, state, action, reward, next_state, done):
        self.buffer.store(state, action, reward, next_state, done)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        self.next_q.train()
        self.eval_q.train()

        states, actions, rewards, next_states, dones = self.buffer.get_random_replays(self.batch_size)

        state_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_state_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        idxs = np.arange(self.batch_size)

        self.optimizer.zero_grad()

        # predict values given the current state-action Q(s, a)
        pred_q = self.eval_q(state_tensor)[idxs, actions]
        # predict values given the next state (for argmax Q(s_t+1, a) later)
        next_q_pred = self.next_q(next_state_tensor)
        next_state_actions_pred = self.eval_q(next_state_tensor)

        # action a that max Q(s_t+1, a)
        max_action = torch.argmax(next_state_actions_pred, dim=1)
        # change the value of the next state to 0 next state is terminal
        next_q_pred[dones] = 0.0

        target = rewards_tensor + self.gamma * next_q_pred[idxs, max_action]
        loss = self.loss_func(pred_q, target)
        loss.backward()
        self.optimizer.step()

        self._change_network_weight()
        self._decrease_epsilon

        self.learn_counter += 1

    def _change_network_weight(self):
        if self.learn_counter % self.replace_next == 0:
            self.next_q.load_state_dict(self.eval_q.state_dict())

    def _decrease_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions_space)
        else:
            self.eval_q.eval()
            with torch.no_grad():
                X = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                actions = self.eval_q(X)
                return torch.argmax(actions).item()

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

    def _get_image_tensor(self):
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
        return state_images

    def step(self, action):
        _, reward, done, _ = self.env.step(action)
        new_state = self._get_image_tensor()

        return new_state, reward, done


    def reset(self):
        self.env.reset()
        state = self._get_image_tensor()

        return state

    def train(self):
        for episode in (t := trange(self.episode)):
            rewards = 0
            n_step = 0
            done = False
            state = self.reset()

            while not done:
                action = self.choose_action(state)
                new_state, reward, done = self.step(action)
                rewards += reward

                self.store_states_action(state, action, reward, new_state, done)
                self.learn()

                state = new_state
                n_step += 1

            self.all_rewards.append(rewards)
            t.set_description(f'Episode: {episode}, reward: {rewards}, total length: {n_step}')

    def play(self):
        self.eval_q.eval()
